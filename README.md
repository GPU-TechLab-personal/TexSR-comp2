# TexSR-comp2

Texture SR models as the initialisation of [PBR-SR](https://github.com/TerenceCYJ/PBR-SR) (NeurIPS 2025). Only the optimisation stage of PBR-SR is run: DiffBIR pseudo-GT and the paper's settings. The init comes from:

- **R1**: [DeepTextures](https://github.com/NethamGH/DeepTextures), retrained on [MatSynth](https://huggingface.co/datasets/gvecchio/MatSynth) with an added MatUp render loss
- **MatUp**: [MatUp](https://github.com/AlbanGauthier/matup_egsr) (EGSR 2024)
- the official PBR-SR init, as the baseline

Every command, in order, is in **[COMMANDS.txt](COMMANDS.txt)** (Git Bash on Windows + Docker Desktop, RTX 50-series). What differs from the earlier runs is in [docs/DEVIATIONS.md](docs/DEVIATIONS.md).

| path | what |
|---|---|
| `external/` | the four methods as git submodules at pinned commits, never edited |
| `docker/` | one image per method (`dataprep`, `deeptex`, `matup`, `pbrsr`) |
| `scripts/setup.sh` | weights, images and a GPU check per image |
| `scripts/r1.sh` | R1: MatSynth pairs → grouped TFRecord → 40k-step training |
| `scripts/init.sh` | Cradle_Globe: PBR-SR's LR maps → R1 / MatUp x4 → the three `read` init PNGs |
| `variants/deeptex/` | the R1 trainer (hooks around upstream's loop) and data builders |
| `variants/pbr_sr/server/` | PBR-SR runs (`run_example.sh`, `status.sh`) |
| `runs/` | everything produced (gitignored) |

The upstream licenses apply to each submodule.
