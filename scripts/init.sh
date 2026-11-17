#!/usr/bin/env bash
# Our SR models as PBR-SR's init on Cradle_Globe: LR export -> R1 / MatUp x4 -> `read` PNGs.
# Run from the repo root in Git Bash:
#
#   bash scripts/init.sh export            PBR-SR's own 1024^2 LR maps + official HiT AO (pbrsr, ~1 min)
#   bash scripts/init.sh r1 [STEP]         R1 x4 of the LR maps (deeptex; STEP 30000|35000|40000, default 40000)
#   bash scripts/init.sh matup             MatUp x4 as released, 100 tiles of 128^2, blended (detached, ~1-2 h)
#   bash scripts/init.sh matup_ft [STEP]   the same with MatUp's SwinIR-L fine-tuned on MatSynth
#                                          (scripts/matup_ft.sh; STEP 30000|35000|40000, default 40000)
#   bash scripts/init.sh pack ARM          ARM = r1 | matup | matup_ft -> runs/pbr_sr/init/read_ARM/
#   bash scripts/init.sh status            what exists, MatUp tile progress
#
# Then the PBR-SR run with that init (same paper preset, no tiling):
#   INIT=r1|matup|matup_ft bash variants/pbr_sr/server/run_example.sh
# Details: variants/pbr_sr/init/*.py (each file's header).
set -euo pipefail
export MSYS_NO_PATHCONV=1

REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
HOSTREPO=$(pwd -W 2>/dev/null || pwd)      # Git Bash: C:/...; Linux: the plain path
mkdir -p runs/logs runs/cache/nv
if docker compose version >/dev/null 2>&1; then DC="docker compose"; else DC="docker-compose"; fi
I=runs/pbr_sr/init
W=/workspace
MATUP_W=external/matup_egsr/pretrained_models/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth

attached() { local svc=$1; shift; ( cd docker && $DC run --rm -T "$svc" "$@" ); }
need() { [ -e "$1" ] || { echo "missing $1 - $2" >&2; exit 1; }; }

# run_matup ARM [docker-run options...]: matup_tiles.py into runs/pbr_sr/init/ARM, detached
run_matup() {
  local arm=$1 name=init_$1; shift
  need $I/lr/lr16/baseColor.png "run: bash scripts/init.sh export"
  if [ -n "$(docker ps -q -f "name=^${name}$")" ]; then echo "$name is already running" >&2; exit 1; fi
  [ -n "$(docker ps -aq -f "name=^${name}$")" ] && docker rm "$name" >/dev/null
  ( cd docker && $DC run -d -T --name "$name" "$@" matup bash -c \
    "python $W/variants/pbr_sr/init/matup_tiles.py --out $W/$I/$arm > $W/runs/logs/$name.log 2>&1; echo EXIT \$? >> $W/runs/logs/$name.log" )
  echo "started $name   log: runs/logs/$name.log   rerun to resume after a stop"
}

case "${1:-}" in
  export)
    attached pbrsr python $W/variants/pbr_sr/init/export_lr.py ;;
  r1)
    need $I/lr/lr8/baseColor.png "run: bash scripts/init.sh export"
    STEP=${2:-40000}
    CK=runs/deeptex/render_r1/checkpoints/esrgan_40k
    # upstream saves every 5000 steps into ckpt-1, ckpt-2, ... and keeps the last 3.
    # R1_CKPT=<checkpoint prefix> overrides (pipeline tests with the smoke checkpoint).
    PREFIX=${R1_CKPT:-$CK/ckpt-$((STEP / 5000))}
    need "$PREFIX.index" "no checkpoint $PREFIX (step $STEP) - train R1 first: bash scripts/r1.sh train"
    attached deeptex python $W/variants/deeptex/infer/deeptex_sr.py \
      --cfg $W/variants/deeptex/render_loss/r1.yaml --ckpt $W/$PREFIX \
      --in-dir $W/$I/lr/lr8 --out-dir $W/$I/r1 --tile-size 256 --overlap 16
    echo "${R1_CKPT:-step $STEP}" > $I/r1/STEP ;;
  matup)
    run_matup matup ;;
  matup_ft)
    # MatUp unchanged; only the SwinIR-L file it loads (sr_swinir.py: ./pretrained_models/...)
    # is replaced, inside this container only, by the fine-tuned EMA weights. The released
    # file on disk is not touched. MATUP_FT_W=<file> overrides (pipeline tests).
    STEP=${2:-40000}
    FT=${MATUP_FT_W:-runs/matup_ft/kair/swinirL_ft/models/${STEP}_E.pth}
    need "$FT" "train it first: bash scripts/matup_ft.sh train"
    mkdir -p $I/matup_ft && echo "$FT" > $I/matup_ft/WEIGHTS
    run_matup matup_ft -v "$HOSTREPO/$FT:$W/$MATUP_W:ro" ;;
  pack)
    ARM=${2:?r1, matup or matup_ft}
    attached dataprep python variants/pbr_sr/init/pack_read.py --arm "$ARM" ;;
  status)
    for f in $I/lr/meta.json $I/r1/baseColor.png $I/matup/baseColor.png $I/matup_ft/baseColor.png \
             $I/read_r1/sr_ks_output.png $I/read_matup/sr_ks_output.png $I/read_matup_ft/sr_ks_output.png; do
      [ -e "$f" ] && echo "ok       $f" || echo "missing  $f"
    done
    [ -f $I/r1/STEP ] && echo "R1 init from: $(cat $I/r1/STEP)"
    [ -f $I/matup_ft/WEIGHTS ] && echo "MatUp-ft SwinIR-L: $(cat $I/matup_ft/WEIGHTS)"
    for arm in matup matup_ft; do
      n=init_$arm
      [ -n "$(docker ps -aq -f "name=^${n}$")" ] || continue
      docker inspect -f "== $n: {{.State.Status}} exit={{.State.ExitCode}}" "$n"
      ok=$(ls $I/$arm/tiles/*/out/*/baseColor_opt.png 2>/dev/null | wc -l)
      echo "tiles done: $ok of 100"
      [ -f runs/logs/$n.log ] && tail -2 runs/logs/$n.log
    done ;;
  *)
    sed -n '2,15p' "$0"; exit 1 ;;
esac
