#!/usr/bin/env bash
# Our SR models as PBR-SR's init on Cradle_Globe: LR export -> R1 / MatUp x4 -> `read` PNGs.
# Run from the repo root in Git Bash:
#
#   bash scripts/init.sh export        PBR-SR's own 1024^2 LR maps + official HiT AO (pbrsr, ~1 min)
#   bash scripts/init.sh r1 [STEP]     R1 x4 of the LR maps (deeptex; STEP 30000|35000|40000, default 40000)
#   bash scripts/init.sh matup         MatUp x4, 100 tiles of 128^2, blended (matup, detached, ~1-2 h)
#   bash scripts/init.sh pack ARM      ARM = r1 | matup -> runs/pbr_sr/init/read_ARM/ (the 3 `read` PNGs)
#   bash scripts/init.sh status        what exists, MatUp tile progress
#
# Then the PBR-SR run with that init (same paper preset, no tiling):
#   INIT=r1    bash variants/pbr_sr/server/run_example.sh
#   INIT=matup bash variants/pbr_sr/server/run_example.sh
# Details: variants/pbr_sr/init/*.py (each file's header).
set -euo pipefail
export MSYS_NO_PATHCONV=1

REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
mkdir -p runs/logs runs/cache/nv
if docker compose version >/dev/null 2>&1; then DC="docker compose"; else DC="docker-compose"; fi
I=runs/pbr_sr/init
W=/workspace

attached() { local svc=$1; shift; ( cd docker && $DC run --rm -T "$svc" "$@" ); }
need() { [ -e "$1" ] || { echo "missing $1 - $2" >&2; exit 1; }; }

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
    need $I/lr/lr16/baseColor.png "run: bash scripts/init.sh export"
    if [ -n "$(docker ps -q -f name=^init_matup$)" ]; then echo "init_matup is already running" >&2; exit 1; fi
    [ -n "$(docker ps -aq -f name=^init_matup$)" ] && docker rm init_matup >/dev/null
    ( cd docker && $DC run -d -T --name init_matup matup bash -c \
      "python $W/variants/pbr_sr/init/matup_tiles.py > $W/runs/logs/init_matup.log 2>&1; echo EXIT \$? >> $W/runs/logs/init_matup.log" )
    echo "started init_matup   log: runs/logs/init_matup.log   rerun to resume after a stop" ;;
  pack)
    ARM=${2:?r1 or matup}
    attached dataprep python variants/pbr_sr/init/pack_read.py --arm "$ARM" ;;
  status)
    for f in $I/lr/meta.json $I/r1/baseColor.png $I/matup/baseColor.png $I/read_r1/sr_ks_output.png $I/read_matup/sr_ks_output.png; do
      [ -e "$f" ] && echo "ok       $f" || echo "missing  $f"
    done
    [ -f $I/r1/STEP ] && echo "R1 init from step $(cat $I/r1/STEP)"
    if [ -n "$(docker ps -aq -f name=^init_matup$)" ]; then
      docker inspect -f '== init_matup: {{.State.Status}} exit={{.State.ExitCode}}' init_matup
      d=$(ls -d $I/matup/tiles/*/ 2>/dev/null | wc -l)
      ok=$(ls $I/matup/tiles/*/out/*/baseColor_opt.png 2>/dev/null | wc -l)
      echo "MatUp tiles done: $ok (of 100; $d started)"
      [ -f runs/logs/init_matup.log ] && tail -3 runs/logs/init_matup.log
    fi ;;
  *)
    sed -n '2,16p' "$0"; exit 1 ;;
esac
