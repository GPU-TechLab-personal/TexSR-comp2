# Deviations from the earlier setup

This repository carries only what the PBR-SR study needs. That is: the DeepTextures + MatUp render-loss arm (R1), MatUp, and the official PBR-SR. The code under `variants/` is copied unchanged from the earlier project. Its paths are kept the same, so every comment and log path still reads the same.

The list below is everything that differs from how R1, MatUp and PBR-SR were run before. All earlier runs were on an RTX 3080 Ti; PBR-SR ran on an RTX 5080.

## 1. GPU: RTX 50-series (Blackwell, sm_120)

| image | earlier | here | why |
|---|---|---|---|
| `deeptex` | TF 2.10.1 / CUDA 11.2 | **same** | TF 2.10.1 has no sm_120 binaries, but its PTX JIT-compiles through the driver. On an RTX 5080 (2026-11-17): conv, gradient and matmul ran correctly. The first call took 401 s, compiling; the next took 1 ms. `CUDA_CACHE_PATH=/workspace/runs/cache/nv` (4 GB) keeps the compiled kernels, so only the first start of a checkout pays. |
| `matup` | torch 2.0.0 / CUDA 11.7 | **torch 2.7.1 / CUDA 12.8** | torch 2.0.0 ships kernels up to sm_86 and no PTX. On the 5080 it prints "sm_120 is not compatible" and no kernel runs. 2.7.1 is the first release with sm_120, and is the same stack as `pbrsr`. The other packages keep the earlier pins: `numpy<2`, `opencv<5`, `timm<0.9`, `requests`. Checked with MatUp's own example `victorian_wood` (`--upscaler swinir`) on the 5080: the loss went 0.0701 → 0.0614 over the 10 rounds, all 5 maps were written, and it took 84 s. |
| `pbrsr` | torch 2.7.1 / CUDA 12.8 | same | |

The JIT path in `deeptex` is a different kernel compilation of the same TF version. Results can differ from the 3080 Ti runs in the last bits, in the same way two GPU models already differ.

## 2. R1 training data: all MatSynth train shards

- **Earlier:** 216 of the 431 train shards were present locally (the rest were git-lfs stubs). That gave 2,904 materials, 14,520 pairs.
- **What was lost:** which 216 shards those were is not recorded anywhere that survived.
- **Here:** all 431 shards are downloaded with `hf download`, and `matsynth_make_pairs.py` uses every real shard. So R1 sees about twice as many materials in the same 40k steps.
- **Unchanged:** the recipe itself (`r1.yaml`), the degradation, and the exclusion of the 12 evaluation materials and their blends.

## 3. `w_render` is not recalibrated

The calibration gate needs trained `matsynth_all` and `matsynth_chratio` checkpoints at 30k/35k/40k. That is two extra 40k runs, and both were lost.

`r1.yaml` keeps `w_render: 10.0`, the value that gate produced on 2026-10-07. With the larger dataset (§2) the gate's ratio could come out somewhat different; this run does not re-measure it.

## 4. Orchestration

The PowerShell orchestrators (`run.ps1`, `lib.ps1`) are not carried over. `scripts/r1.sh` runs the same Python entry points with the same arguments as `run.ps1 -Stage prep-matsynth / prep-grouped / render-train`. The following are dropped:
- **The checkpoint keeper (`-KeepAllCheckpoints`):** upstream keeps the last 3 checkpoints. With `save_steps: 5000` those are 30k / 35k / 40k, which are the steps used before.
- **The VRAM pre-check (`Assert-RenderVram`):** it targeted a 12 GB card.
- **The failure classifier:** read the log's last lines instead (`bash scripts/r1.sh status`).

## 5. Our SR maps as PBR-SR's init (`scripts/init.sh`, `variants/pbr_sr/init/`)

- **Input = PBR-SR's own LR.** `export_lr.py` calls `load_mesh_data(data_dir, downsample_scale=4)`. That is the call `run_pbr_sr.py:708` makes, so both models see exactly the 1024^2 maps the official run super-resolves.
- **AO = the official one.** It is HiT-SRF x4 on the LR ks, built and applied the way `run_pbr_sr.py:184-188, 1366` does it. Channel 0 is already 1 - AO, which is what `read` expects.
- **8-bit `read` files.** `read` loads with `imageio.imread / 255` (`nvdiff_render/util.py:442`), so the three files must be 8-bit. The official inits stay float `.pt`. Our arms are therefore quantised to 8 bits, including the HiT AO channel they share with the official run.
- **MatUp is run in tiles.** Its SwinIR-L step upsamples whole renders (`sr_swinir.py`: `tile = None`), which does not fit at 1024^2. So the work is split:
  - 128^2 tiles (MatUp's `--material_size`), stride 112, on an 8 px reflect-padded LR.
  - 16 HR px are cropped per side, to remove MatUp's 8 px zero border and its wrap-around bilinear init at the tile edge.
  - Neighbours are blended over a 32 px linear ramp.
  - 100 tiles, ~75 s each on an RTX 5080.
- **MatUp inputs.**
  - Height is a constant 0.5, which means no displacement: Cradle_Globe has no height map.
  - Normals are read and written in MatUp's internal convention (no `--ogl_normals`), so the output keeps Cradle_Globe's own convention.
- **R1 inputs.** The 8-bit LR maps go through `deeptex_sr.py`, tiled 256 LR px with 16 overlap. For a grey map, the mean of the three output channels is used.
