#!/usr/bin/env bash
# Start the official PBR-SR example (Cradle_Globe, README usage) in a DETACHED
# container, so closing or losing the SSH session does not stop it.
#
#   bash variants/pbr_sr/server/run_example.sh [RUN_NAME] [official|tiled] [paper|readme]
#     RUN_NAME  default cradle_globe_<preset>; names the container (pbrsr_<RUN_NAME>),
#               runs/pbr_sr/<RUN_NAME>/ and runs/pbr_sr/logs/<RUN_NAME>.*
#     official  configs/ of the submodule as shipped, no tiling (default). Needs
#               ~26 GB VRAM: measured 15.3 GB dedicated + 11 GB spilled on the 16 GB
#               5080, so it fits the 32 GB 5090 (not yet measured there).
#     tiled     variants/pbr_sr/configs_tiled (only the 4 DiffBIR *_tiled flags on);
#               what fits in 16 GB, ~31 s per pseudo-GT view on the 5080. Not used for
#               the experiments (decided 2026-11-16: no tiling).
#     paper     the paper's stated settings (default) - see PRESET below
#     readme    the README usage example as written (n_iter 10000, diffbir_steps 10)
#   N_ITER=... overrides n_iter (e.g. a short test).  DRY_RUN=1 prints the command only.
#   INIT=deeptex_renloss_21|matup|matup_ft  start from our SR maps instead of the official init: the three
#               files of runs/pbr_sr/init/read_<INIT>/ (scripts/init.sh) are put in
#               the run's data dir and --kd/ks/n_init_method are 'read'. Everything
#               else (pseudo-GT, preset, losses) is unchanged. Default RUN_NAME gets
#               the suffix _<INIT>.
#   VIEWS_TEST=1  pipeline test only: --train_elev_index 1 (50 views, not the paper's 750)
#   SSIM_W=10   paper preset only: --pbr_interpolate_ssim_weight (default 100, see below).
#               Any other value adds _ssim<SSIM_W> to the default RUN_NAME, so the baseline
#               is not overwritten. Ablation for the 4-texel mosaic seen after PBR-SR.
#   CACHE_FROM=<RUN>  read the pseudo-GT from runs/pbr_sr/<RUN>/cache_render instead of
#               building it (~57 min saved, and an identical pseudo-GT). Only valid when
#               <RUN> has the SAME init: the cache file name does not encode the init
#               (run_pbr_sr.py:2085), so another arm's .pt would load silently. If the .pt is
#               missing there, it is built and written into that folder.
#
# PRESET paper - every value the paper states, mapped onto the official arguments
# (PBR-SR, NeurIPS 2025, Sec. 3.3/3.5, Implementation Details, App. A.2). Arguments
# whose code default already equals the paper are left at the default:
#   paper                                   code default   set here
#   2000 iterations                         n_iter 10000   --n_iter 2000
#   DiffBIR denoising steps "five"          10             --diffbir_steps 5
#   init (Sec. 3.2): albedo by the SR       ks: CAMixer +  --ks_init_method 'US_bicubic'
#     model (DiffBIR), ARM and normal by    HiT-SRF x4     (kd DiffBIR_x4 and n US_bicubic
#     bicubic interpolation                                are already the defaults)
#   PBR consistency w_Km (metallic) 0.1     1.0 (l1, ssim) --matellic_weight_l1 0.1
#                                                          --matellic_weight_interpolate_ssim 0.1
#   SSIM term, Eq. 6 inside l_pbr L_pbr:    10 (normal     --pbr_interpolate_ssim_weight 100
#     l_pbr x w_T x lambda_ssim             x10 = 100)     --n_weight_interpolate_ssim 1.0
#     = 10 x 1 x 10 = 100 (w_Km: 10)                       (decided 2026-11-18: Eq. 6 as
#                                                          written; the released code gives 10)
#   L_pbr sums T in {Kd, Kr, Km, Kn}:       AO l1 1.0,     --occlusion_weight_l1 0
#     no AO term                            AO ssim 1.0    --occlusion_weight_interpolate_ssim 0
#   L_total = L_robust + l_pbr L_pbr        patch-consist. --pbr_patch_consistency_weight 0
#             + l_tv L_tv (no other term)   on, w 1.0      (store_true default True: only
#                                                          its weight can switch it off)
#   already equal: 
#       Adam lr 1e-4 constant (StepLR step 10000 never fires in 2000 its),
#       batch 4 views, 750 views = 15 elevations x 50, FoV 10, camera distance 3.25,
#       pseudo-GT 1024^2 (render_res 512 x cam_big_upscale 2), robust lambda_pix 100 / lambda_reg 0.5, 
#       weight map 64^2, l_pbr 10 (pbr_loss_weight_l1), l_tv 0.5, w_Kd = w_Kr = 1.0.
#
#   NOT matched (no argument for them; the official code is not edited):
#       - evaluation views: the paper evaluates on 240 views (6 elevations x 40; angles, FoV
#         and distance not stated). The code renders two fixed sets of 5 elevations x 20 =
#         100 views each, 512^2 at distance 3.25, FoV 5 (set1) and FoV 10 (set2)
#         (run_pbr_sr.py:759-773; --circle_n_view_val is not used and --val_elev is
#         type=list, so neither changes them from the command line).
#       - Pool in the SSIM term: Eq. 6 average-pools T_theta for both L1 and SSIM; the code
#         average-pools for L1 but resizes bicubically for SSIM (run_pbr_sr.py:4245-4275).
#       - AO in L_tv: the TV term runs over kd, AO, roughness, metallic and normal together
#         (run_pbr_sr.py:4351-4355), so AO keeps a TV term (weight 0.5) although it is out
#         of L_pbr. Only --pbr_tv_loss / --tv_weight exist, and they switch TV off for every
#         map, so it is left on.
#
#   Ground truth: Cradle_Globe's shipped textures are 4096^2 (metal/rough 2048^2); the
#   official loader treats them as the HR ground truth and builds the 1024^2 LR input
#   itself (pbr_sr/utils_online.py:402-477), so the texture metrics are against them.
#
# Then: bash variants/pbr_sr/server/status.sh RUN_NAME
# Stop: docker stop pbrsr_RUN_NAME
#
# Every run gets its own copy of the bundled example: the official loader writes its
# init caches (sr_kd_output_*.pt) next to the input, and they depend on the DiffBIR
# config, so an official and a tiled run must not share an input folder. The path
# keeps .../LR_mesh/Cgtrader/<mesh> because the official loader picks its branch by
# that substring (pbr_sr/utils_online.py; otherwise it stops in pdb).
set -euo pipefail
export MSYS_NO_PATHCONV=1

REPO=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$REPO"
CONFIG=${2:-official}
PRESET=${3:-paper}
INIT=${INIT:-}
SSIM_W=${SSIM_W:-100}
[ "$PRESET" = paper ] || [ "$SSIM_W" = 100 ] || { echo "SSIM_W only applies to the paper preset" >&2; exit 1; }
SSIM_SUFFIX=; [ "$SSIM_W" = 100 ] || SSIM_SUFFIX=_ssim$SSIM_W
RUN=${1:-cradle_globe_$PRESET${INIT:+_$INIT}$SSIM_SUFFIX}
CACHE_RUN=${CACHE_FROM:-$RUN}
if [ -n "${CACHE_FROM:-}" ] && ! ls runs/pbr_sr/$CACHE_FROM/cache_render/*.pt >/dev/null 2>&1; then
  echo "note: no .pt in runs/pbr_sr/$CACHE_FROM/cache_render - the pseudo-GT will be built (and saved there)" >&2
fi
NAME=pbrsr_$RUN
case "$CONFIG" in
  official) CFG_ENV=() ;;
  tiled)    CFG_ENV=(-e PBR_SR_CONFIG_DIR=/workspace/variants/pbr_sr/configs_tiled) ;;
  *) echo "config must be official or tiled, got: $CONFIG" >&2; exit 1 ;;
esac
case "$PRESET" in
  paper)  N_ITER=${N_ITER:-2000}
          KS_INIT=US_bicubic
          EXTRA="--diffbir_steps 5 \
 --matellic_weight_l1 0.1 --matellic_weight_interpolate_ssim 0.1 \
 --pbr_interpolate_ssim_weight $SSIM_W --n_weight_interpolate_ssim 1.0 \
 --occlusion_weight_l1 0 --occlusion_weight_interpolate_ssim 0 \
 --pbr_patch_consistency_weight 0" ;;
  readme) N_ITER=${N_ITER:-10000}
          KS_INIT=CAMixer_HiT_SR_x4
          EXTRA="--diffbir_steps 10" ;;
  *) echo "preset must be paper or readme, got: $PRESET" >&2; exit 1 ;;
esac
case "$INIT" in
  "")        INIT_ARGS="--kd_init_method 'DiffBIR_x4' --ks_init_method '$KS_INIT' --n_init_method 'US_bicubic'" ;;
  deeptex_renloss_21|matup|matup_ft)  INIT_DIR=runs/pbr_sr/init/read_$INIT
             for f in sr_kd_output.png sr_ks_output.png sr_n_output.png; do
               [ -f "$INIT_DIR/$f" ] || { echo "missing $INIT_DIR/$f - run: bash scripts/init.sh pack $INIT" >&2; exit 1; }
             done
             INIT_ARGS="--kd_init_method 'read' --ks_init_method 'read' --n_init_method 'read'" ;;
  *) echo "INIT must be deeptex_renloss_21, matup or matup_ft (or unset), got: $INIT" >&2; exit 1 ;;
esac
[ "${VIEWS_TEST:-0}" = 1 ] && EXTRA="$EXTRA --train_elev_index 1"
if docker compose version >/dev/null 2>&1; then DC="docker compose"; else DC="docker-compose"; fi

SRC=external/PBR-SR/data/LR_mesh/Cgtrader/Cradle_Globe
DATA=runs/pbr_sr/$RUN/data/LR_mesh/Cgtrader/Cradle_Globe
LOGS=runs/pbr_sr/logs
W=/workspace

if [ -n "$(docker ps -aq -f "name=^${NAME}$")" ]; then
  if [ "$(docker inspect -f '{{.State.Running}}' "$NAME")" = true ]; then
    echo "$NAME is already running: bash variants/pbr_sr/server/status.sh $RUN" >&2; exit 1
  fi
  [ "${DRY_RUN:-0}" = 1 ] || docker rm "$NAME" >/dev/null
fi

PY="python run_pbr_sr.py \
 --data_dir '$W/$DATA' \
 --out_dir '$W/runs/pbr_sr/$RUN/outputs' \
 --n_iter $N_ITER \
 $INIT_ARGS \
 --pbr_sr_scale 4 \
 --img_sr_model 'DiffBIR_x4' \
 --render_cache_dir '$W/runs/pbr_sr/$CACHE_RUN/cache_render' \
 --robust_opti \
 $EXTRA"
L=$W/$LOGS/$RUN
INNER="{ echo CONFIG=$CONFIG CONFIG_DIR=\${PBR_SR_CONFIG_DIR:-official configs/}; nvidia-smi -L; \
 time $PY; } > $L.log 2>&1; echo \"EXIT \$?\" >> $L.log; date +%FT%T > $L.end"

if [ "${DRY_RUN:-0}" = 1 ]; then
  echo "(cd docker && $DC run -d -T --name $NAME ${CFG_ENV[*]} pbrsr bash -c \"$INNER\")"; exit 0
fi

mkdir -p "$(dirname "$DATA")" "$LOGS"
[ -f "$DATA/mesh.obj" ] || cp -r "$SRC" "$(dirname "$DATA")/"
# `read` takes the first file in data_dir whose name contains sr_{kd,ks,n}_output.png
[ -n "$INIT" ] && cp "$INIT_DIR"/sr_kd_output.png "$INIT_DIR"/sr_ks_output.png "$INIT_DIR"/sr_n_output.png "$DATA/"
[ -f "$LOGS/$RUN.log" ] && mv "$LOGS/$RUN.log" "$LOGS/$RUN.prev_$(date +%Y%m%d_%H%M%S).log"
rm -f "$LOGS/$RUN.end"
{ echo "start $(date +%FT%T) host=$(hostname) config=$CONFIG preset=$PRESET init=${INIT:-official} n_iter=$N_ITER ssim_w=$SSIM_W cache=$CACHE_RUN"
  echo "$PY"; } > "$LOGS/$RUN.start"

( cd docker && $DC run -d -T --name "$NAME" ${CFG_ENV[@]+"${CFG_ENV[@]}"} pbrsr bash -c "$INNER" )

# GPU sampler on the host, detached like the run; stops when the container stops.
SETSID=$(command -v setsid || true)   # absent in Git Bash; Linux has it
nohup $SETSID bash -c "
  nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu,power.draw,temperature.gpu \
    --format=csv,noheader -l 5 >> '$LOGS/${RUN}_gpu.csv' & s=\$!
  while [ \"\$(docker inspect -f '{{.State.Running}}' '$NAME' 2>/dev/null)\" = true ]; do sleep 30; done
  kill \$s" > /dev/null 2>&1 < /dev/null &

echo "started $NAME  (config=$CONFIG, preset=$PRESET, n_iter=$N_ITER)"
echo "status: bash variants/pbr_sr/server/status.sh $RUN"
