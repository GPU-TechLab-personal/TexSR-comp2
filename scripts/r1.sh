#!/usr/bin/env bash
# DeepTextures + MatUp render loss, arm R1: MatSynth -> pairs -> grouped TFRecord -> training.
# Run from the repo root in Git Bash (or any Linux shell):
#
#   bash scripts/r1.sh evalset    12 evaluation materials from test-00000 (excluded from training)
#   bash scripts/r1.sh check      training-pair degradation == evaluation degradation, pixel for pixel
#   bash scripts/r1.sh pairs      MatSynth train shards -> 512/128 PNG pairs        (detached, hours)
#   bash scripts/r1.sh grouped    pairs -> grouped TFRecord, then byte-verify it    (detached)
#   bash scripts/r1.sh smoke      200 training steps into runs/deeptex/r1_smoke     (detached, ~15 min)
#   bash scripts/r1.sh train      the full 40k-step R1 run                          (detached, hours)
#   bash scripts/r1.sh status     what is running, log tails, checkpoints, GPU
#   bash scripts/r1.sh stop STAGE stop a detached stage
#
# Detached stages run in a container named r1_<stage>; closing the terminal or the
# remote-desktop window does not stop them. Logs: runs/logs/r1_<stage>.log (EXIT code
# on the last line when it ends). Rerunning pairs / grouped / train continues where
# they stopped (pairs: per-material markers; train: resumes from its last checkpoint).
#
# The recipe is variants/deeptex/render_loss/r1.yaml, unchanged (w_render 10.0 from
# the 2026-10-07 calibration gate - see docs/DEVIATIONS.md for why the gate is not rerun).
set -euo pipefail
export MSYS_NO_PATHCONV=1

REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
LOGS=runs/logs
mkdir -p "$LOGS" runs/cache/nv
if docker compose version >/dev/null 2>&1; then DC="docker compose"; else DC="docker-compose"; fi

PAIRS=/workspace/runs/deeptex/datasets/matsynth
GROUPED=/workspace/runs/deeptex/datasets/matsynth_grouped/matsynth_grouped_lrx4.tfrecord
CFG=/workspace/variants/deeptex/render_loss/r1.yaml
TRAIN="python /workspace/variants/deeptex/paper_loop/main.py --experiment render --cfg_path $CFG \
 --stage esrgan --losses upstream --shuffle off --allow-dead-lr-schedule --pretrain-dir none \
 --no-val --log-every 100 --render-monitor-every 100"

attached() {  # service, command...
  local svc=$1; shift
  ( cd docker && $DC run --rm -T "$svc" "$@" )
}

detached() {  # stage, service, shell command
  local stage=$1 svc=$2 cmd=$3 name=r1_$1
  if [ -n "$(docker ps -q -f "name=^${name}$")" ]; then
    echo "$name is already running - bash scripts/r1.sh status" >&2; exit 1
  fi
  [ -n "$(docker ps -aq -f "name=^${name}$")" ] && docker rm "$name" >/dev/null
  [ -f "$LOGS/$name.log" ] && mv "$LOGS/$name.log" "$LOGS/$name.prev_$(date +%Y%m%d_%H%M%S).log"
  local L=/workspace/$LOGS/$name
  ( cd docker && $DC run -d -T --name "$name" "$svc" bash -c \
      "{ date +%FT%T; nvidia-smi -L 2>/dev/null; $cmd; } > $L.log 2>&1; echo \"EXIT \$?\" >> $L.log" )
  echo "started $name   log: $LOGS/$name.log   status: bash scripts/r1.sh status"
}

need() { [ -e "$1" ] || { echo "missing $1 - $2" >&2; exit 1; }; }

case "${1:-}" in
  evalset)
    attached dataprep python variants/util/matsynth_extract.py --index 0 --count 12 ;;
  check)
    need runs/matsynth "run: bash scripts/r1.sh evalset"
    attached dataprep python variants/deeptex/matsynth/matsynth_make_pairs.py --check-eval ;;
  pairs)
    need runs/matsynth "run: bash scripts/r1.sh evalset"
    detached pairs dataprep "python variants/deeptex/matsynth/matsynth_make_pairs.py --out $PAIRS --jobs 4" ;;
  grouped)
    need runs/deeptex/datasets/matsynth/_meta "run: bash scripts/r1.sh pairs (and let it finish)"
    detached grouped deeptex "python /workspace/variants/deeptex/render_loss/build/texsr_build_grouped_tfrecord.py \
 --root $PAIRS --out $GROUPED && python /workspace/variants/deeptex/render_loss/build/texsr_verify_grouped.py \
 --tfrecord $GROUPED --root $PAIRS" ;;
  smoke)
    need runs/deeptex/datasets/matsynth_grouped/verify_report.json "run: bash scripts/r1.sh grouped"
    detached smoke deeptex "$TRAIN --max-steps 200 --save-steps 100 \
 --checkpoint-root /workspace/runs/deeptex/r1_smoke/checkpoints --log-root /workspace/runs/deeptex/r1_smoke/logs" ;;
  train)
    need runs/deeptex/datasets/matsynth_grouped/verify_report.json "run: bash scripts/r1.sh grouped"
    detached train deeptex "$TRAIN" ;;
  stop)
    docker stop "r1_${2:?stage name, e.g. train}" ;;
  status)
    for s in pairs grouped smoke train; do
      n=r1_$s
      [ -n "$(docker ps -aq -f "name=^${n}$")" ] || continue
      docker inspect -f "== $n: {{.State.Status}} exit={{.State.ExitCode}} oom={{.State.OOMKilled}} started={{.State.StartedAt}}" "$n"
      [ -f "$LOGS/$n.log" ] && tr '\r' '\n' < "$LOGS/$n.log" | grep -v '^\s*$' | tail -"${LINES:-6}" | cut -c1-200
    done
    [ -d runs/deeptex/datasets/matsynth/_meta ] && echo "pairs: $(ls runs/deeptex/datasets/matsynth/_meta | wc -l) materials done"
    for d in runs/deeptex/r1_smoke/checkpoints/esrgan_40k runs/deeptex/render_r1/checkpoints/esrgan_40k; do
      [ -d "$d" ] && echo "checkpoints $d: $(ls "$d" | grep -o 'ckpt-[0-9]*' | sort -u | tr '\n' ' ')"
    done
    nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader ;;
  *)
    sed -n '2,20p' "$0"; exit 1 ;;
esac
