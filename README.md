# TexSR-comp2

Texture SR models as the initialisation of [PBR-SR](https://github.com/TerenceCYJ/PBR-SR) (NeurIPS 2025). Only the optimisation stage of PBR-SR is run: DiffBIR pseudo-GT and the paper's settings. The init comes from:

- **deeptex_renloss_21**: [DeepTextures](https://github.com/NethamGH/DeepTextures), retrained on [MatSynth](https://huggingface.co/datasets/gvecchio/MatSynth) with an added MatUp render loss
- **MatUp**: [MatUp](https://github.com/AlbanGauthier/matup_egsr) (EGSR 2024), as released and with its SwinIR-L upsampler fine-tuned on MatSynth ([KAIR](https://github.com/cszn/KAIR))
- the official PBR-SR init, as the baseline

Every command, in order, is in **[COMMANDS.txt](COMMANDS.txt)** (Git Bash on Windows + Docker Desktop, RTX 50-series). What differs from the earlier runs is in [docs/DEVIATIONS.md](docs/DEVIATIONS.md).

| path | what |
|---|---|
| `external/` | the four methods and KAIR (the MatUp fine-tune trainer) as git submodules at pinned commits, never edited |
| `docker/` | one image per method (`dataprep`, `deeptex`, `matup`, `pbrsr`) |
| `scripts/setup.sh` | lab-PC environment: weights, images and a GPU check, all or per part (`data`, `deeptex_renloss_21`, `matup`, `pbrsr`) |
| `scripts/data.sh` | MatSynth → the evaluation set (`runs/matsynth/`) and the 512/128 pairs every arm shares (`runs/datasets/matsynth/`); no model code |
| `scripts/deeptex_renloss_21.sh` | deeptex_renloss_21: shared pairs → grouped TFRecord → 40k-step training |
| `scripts/matup_ft.sh` | MatUp's SwinIR-L x4 fine-tuned on the same MatSynth pairs (KAIR, comp1's `swinir_cls_ft` recipe) |
| `scripts/init.sh` | Cradle_Globe: PBR-SR's LR maps → deeptex_renloss_21 / MatUp x4 → the three `read` init PNGs |
| `variants/deeptex/` | the deeptex_renloss_21 trainer (hooks around upstream's loop) and data builders |
| `variants/pbr_sr/server/` | PBR-SR runs (`run_example.sh`, `status.sh`) |
| `runs/` | everything produced (gitignored) |

The upstream licenses apply to each submodule.
