# Deviations from the earlier setup

This repository carries only what the PBR-SR study needs. That is: the DeepTextures + MatUp render-loss arm (deeptex_renloss_21), MatUp, and the official PBR-SR. The render-loss arm is named after its loss: DeepTextures retrained with MatUp's l2,1 render loss (`matup/train.py:17-21`, ported in `variants/deeptex/render_loss/render_tf.py`). comp1 called it R1; here the script, config, run folders, container names and init arm all use `deeptex_renloss_21`, and so do the comments of the copied code. The code under `variants/` is otherwise copied unchanged from the earlier project. Its paths are kept the same, so every comment and log path still reads the same.

The list below is everything that differs from how deeptex_renloss_21, MatUp and PBR-SR were run before. All earlier runs were on an RTX 3080 Ti; PBR-SR ran on an RTX 5080.

## 1. GPU: RTX 50-series (Blackwell, sm_120)

| image | earlier | here | why |
|---|---|---|---|
| `deeptex` | TF 2.10.1 / CUDA 11.2 | **same** | TF 2.10.1 has no sm_120 binaries, but its PTX JIT-compiles through the driver. On an RTX 5080 (2026-11-17): conv, gradient and matmul ran correctly. The first call took 401 s, compiling; the next took 1 ms. `CUDA_CACHE_PATH=/workspace/runs/cache/nv` (4 GB) keeps the compiled kernels, so only the first start of a checkout pays. |
| `matup` | torch 2.0.0 / CUDA 11.7 | **torch 2.7.1 / CUDA 12.8** | torch 2.0.0 ships kernels up to sm_86 and no PTX. On the 5080 it prints "sm_120 is not compatible" and no kernel runs. 2.7.1 is the first release with sm_120, and is the same stack as `pbrsr`. The other packages keep the earlier pins: `numpy<2`, `opencv<5`, `timm<0.9`, `requests`. Checked with MatUp's own example `victorian_wood` (`--upscaler swinir`) on the 5080: the loss went 0.0701 → 0.0614 over the 10 rounds, all 5 maps were written, and it took 84 s. |
| `pbrsr` | torch 2.7.1 / CUDA 12.8 | same | |

The JIT path in `deeptex` is a different kernel compilation of the same TF version. Results can differ from the 3080 Ti runs in the last bits, in the same way two GPU models already differ.

## 2. deeptex_renloss_21 training data: all MatSynth train shards

- **Earlier:** 216 of the 431 train shards were present locally (the rest were git-lfs stubs). That gave 2,904 materials, 14,520 pairs.
- **What was lost:** which 216 shards those were is not recorded anywhere that survived.
- **Here:** all 431 shards are downloaded with `hf download`, and `matsynth_make_pairs.py` uses every real shard. So deeptex_renloss_21 sees about twice as many materials in the same 40k steps.
- **Unchanged:** the recipe itself (`deeptex_renloss_21.yaml`), the degradation, and the exclusion of the 12 evaluation materials and their blends.

## 3. `w_render` is not recalibrated

The calibration gate needs trained `matsynth_all` and `matsynth_chratio` checkpoints at 30k/35k/40k. That is two extra 40k runs, and both were lost.

`deeptex_renloss_21.yaml` keeps `w_render: 10.0`, the value that gate produced on 2026-10-07. With the larger dataset (§2) the gate's ratio could come out somewhat different; this run does not re-measure it.

## 4. Orchestration

The PowerShell orchestrators (`run.ps1`, `lib.ps1`) are not carried over. `scripts/data.sh` and `scripts/deeptex_renloss_21.sh` run the same Python entry points with the same arguments as `run.ps1 -Stage prep-matsynth` (data.sh) and `prep-grouped / render-train` (deeptex_renloss_21.sh). The data stage is its own script because deeptex_renloss_21 and the MatUp fine-tune both read its pairs; for the same reason the pairs live at `runs/datasets/matsynth/`, not under `runs/deeptex/` (the scripts pass that path explicitly; the copied code's defaults are unchanged). The following are dropped:
- **The checkpoint keeper (`-KeepAllCheckpoints`):** upstream keeps the last 3 checkpoints. With `save_steps: 5000` those are 30k / 35k / 40k, which are the steps used before.
- **The VRAM pre-check (`Assert-RenderVram`):** it targeted a 12 GB card.
- **The failure classifier:** read the log's last lines instead (`bash scripts/deeptex_renloss_21.sh status`).

## 5. Our SR maps as PBR-SR's init (`scripts/init.sh`, `variants/pbr_sr/init/`)

- **Input = PBR-SR's own LR.** `export_lr.py` calls `load_mesh_data(data_dir, downsample_scale=4)`. That is the call `run_pbr_sr.py:708` makes, so both models see exactly the 1024^2 maps the official run super-resolves.
- **AO = the paper's ARM init, bicubic.** The paper initialises ARM and normal by bicubic interpolation (Sec. 3.2), so the AO channel is the LR ks upsampled x4 by `torch.nn.Upsample(mode='bicubic')`, as the official `US_bicubic` ks init does (`run_pbr_sr.py:858, 1185`). The baseline run uses that same init (§7), so every arm starts from the same AO. Channel 0 is already 1 - AO, which is what `read` expects. Until 2026-11-18 this was HiT-SRF x4, the README's `CAMixer_HiT_SR_x4`.
- **8-bit `read` files.** `read` loads with `imageio.imread / 255` (`nvdiff_render/util.py:442`), so the three files must be 8-bit. The official inits stay float `.pt`. Our arms are therefore quantised to 8 bits, including the bicubic AO channel; the baseline keeps it in float.
- **MatUp is run in tiles.** Its SwinIR-L step upsamples whole renders (`sr_swinir.py`: `tile = None`), which does not fit at 1024^2. So the work is split:
  - 128^2 tiles (MatUp's `--material_size`), stride 112, on an 8 px reflect-padded LR.
  - 16 HR px are cropped per side, to remove MatUp's 8 px zero border and its wrap-around bilinear init at the tile edge.
  - Neighbours are blended over a 32 px linear ramp.
  - 100 tiles, ~75 s each on an RTX 5080.
- **MatUp inputs.**
  - Height is a constant 0.5, which means no displacement: Cradle_Globe has no height map.
  - Normals are read and written in MatUp's internal convention (no `--ogl_normals`), so the output keeps Cradle_Globe's own convention.
- **deeptex_renloss_21 inputs.** The 8-bit LR maps go through `deeptex_sr.py`, tiled 256 LR px with 16 overlap. For a grey map, the mean of the three output channels is used.

## 6. MatUp fine-tuned on MatSynth (`scripts/matup_ft.sh`, `variants/matup_ft/`)

- **What is tuned.** MatUp has no trained weights of its own. It optimises each material, and its only network is the SwinIR-L x4 GAN that upscales the material's renderings (`sr_swinir.py`). That network is what gets fine-tuned. Its architecture is fixed in MatUp's code (`task real_sr`, `--large_model`), so the fine-tune keeps it exactly.
- **Recipe = TexSR-comp1's better SwinIR arm, `swinir_cls_ft`.** In comp1 (`runs/montages/sr_finetune/summary.md`), it was the best SwinIR arm on map PSNR / SSIM / normal angle for all five maps, and on render FLIP. That recipe is:
  - KAIR's official classical trainer (`main_train_psnr.py`, L1, EMA 0.999);
  - fixed area pairs, H patch 256;
  - batch 4, Adam 1e-4, MultiStepLR [20k, 30k] x 0.5;
  - 40k iterations, seed 0.
  `variants/matup_ft/swinirL_ft.json` is comp1's `swinir_cls_ft.json` with only the network (SwinIR-L real, as MatUp builds it) and the starting weights changed.
- **Starting weights:** MatUp's own `003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth` (`params_ema`), so step 0 is MatUp as released.
- **Data:** the shared MatSynth pairs deeptex_renloss_21 also trains on (`scripts/data.sh pairs`), five maps 1:1, read in place. KAIR's option parser only takes a string for `dataroot_*`, so `train.py` wraps `utils_option.parse` to pass the five folders. KAIR itself is not edited.
- **Data domain:** like comp1, the network is tuned on maps, while MatUp applies it to renderings.
- **Image:** the `matup` image (torch 2.7.1) runs KAIR. comp1 ran it on torch 2.0.0, which has no sm_120 kernels.
- **Use in MatUp:** `init.sh matup_ft` bind-mounts `<step>_E.pth` over the released file inside that container only. Neither MatUp's code nor the file on disk changes.
  - Checked on an RTX 5080: a 20-step run saved `10_E` and `20_E`.
  - Inside the container, MatUp's path had `20_E`'s sha256, and `define_model` loaded it with `strict=True` (28.0M parameters).
  - MatUp then ran its example (`victorian_wood`) to the end.

## 7. PBR-SR run settings: the paper's, not the released code's defaults (`variants/pbr_sr/server/run_example.sh`, preset `paper`)

The `paper` preset (the default) sets every value the paper states. The table in the script's header maps each one to the official argument. Changed from the code's defaults:
- 2000 iterations, 5 DiffBIR steps, `w_Km` 0.1, no patch-consistency term (as before).
- **Init (Sec. 3.2):** albedo by DiffBIR, ARM and normal bicubic. The code's default ks init is CAMixerSR + HiT-SRF x4; the preset passes `--ks_init_method US_bicubic`.
- **SSIM weight 100 (Eq. 6 as written, decided 2026-11-18):** `L_total` multiplies `L_pbr` by `lambda_pbr` = 10, and `L_pbr` holds `lambda_ssim` = 10 inside, so SSIM gets 100 per map (metallic 10). The released code gives 10.
- **No AO term in `L_pbr`:** Eq. 6 sums over {Kd, Kr, Km, Kn}. The code also puts AO in the L1 and SSIM terms; the preset sets both AO weights to 0. AO still has a TV term: `L_tv` runs over kd, AO, roughness, metallic and normal together (`run_pbr_sr.py:4351-4355`), and no argument removes one map from it, so AO is regularised by TV (weight 0.5) though it has no data term.

Not matched, because the code has no argument for them and is not edited:
- **Evaluation views.** The paper evaluates on 240 views (6 elevations x 40). The code renders two fixed sets of 5 elevations x 20 = 100 views each, 512^2, FoV 5 and FoV 10 (`run_pbr_sr.py:759-773`). Undecided (2026-11-18).
- **Pool in the SSIM term.** Eq. 6 average-pools for SSIM as for L1; the code resizes bicubically for SSIM.
