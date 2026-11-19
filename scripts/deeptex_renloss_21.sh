#!/usr/bin/env bash
# DeepTextures + MatUp render loss, arm deeptex_renloss_21: shared MatSynth pairs -> grouped TFRecord -> training.
# The name: DeepTextures retrained with MatUp's l2,1 render loss (matup/train.py:17-21, ported in
# variants/deeptex/render_loss/render_tf.py). comp1 called this arm R1.
# The pairs come from scripts/data.sh (evalset, check, pairs); this script only reads them.
# Run from the repo root in Git Bash (or any Linux shell):
#
#   bash scripts/deeptex_renloss_21.sh grouped     pairs -> grouped TFRecord, then byte-verify it    (detached)
#   bash scripts/deeptex_renloss_21.sh smoke       200 training steps into runs/deeptex/deeptex_renloss_21_smoke     (detached, ~15 min)
#   bash scripts/deeptex_renloss_21.sh train       the full 40k-step deeptex_renloss_21 run                          (detached, hours)
#   bash scripts/deeptex_renloss_21.sh status      what is running, log tails, checkpoints, GPU
#   bash scripts/deeptex_renloss_21.sh stop STAGE  stop a detached stage (grouped / smoke / train)
#
# Detached stages run in a container named deeptex_renloss_21_<stage>; closing the terminal or the
# remote-desktop window does not stop them. Logs: runs/logs/deeptex_renloss_21_<stage>.log (EXIT code
# on the last line when it ends). Rerunning train continues from its last checkpoint.
#
# The recipe is variants/deeptex/render_loss/deeptex_renloss_21.yaml, unchanged (w_render 10.0 from
# the 2026-10-07 calibration gate - see docs/DEVIATIONS.md for why the gate is not rerun).
set -euo pipefail
export MSYS_NO_PATHCONV=1

REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
LOGS=runs/logs
mkdir -p "$LOGS" runs/cache/nv
if docker compose version >/dev/null 2>&1; then DC="docker compose"; else DC="docker-compose"; fi

PAIRS=/workspace/runs/datasets/matsynth          # scripts/data.sh pairs
GROUPED=/workspace/runs/deeptex/datasets/matsynth_grouped/matsynth_grouped_lrx4.tfrecord
CFG=/workspace/variants/deeptex/render_loss/deeptex_renloss_21.yaml
TRAIN="python /workspace/variants/deeptex/paper_loop/main.py --experiment render --cfg_path $CFG \
		--stage esrgan --losses upstream --shuffle off --allow-dead-lr-schedule --pretrain-dir none \
 		--no-val --log-every 100 --render-monitor-every 100"

attached() {  # service, command...
  local svc=$1; shift
  ( cd docker && $DC run --rm -T "$svc" "$@" )
}

detached() {  # stage, service, shell command
  local stage=$1 svc=$2 cmd=$3 name=deeptex_renloss_21_$1
  if [ -n "$(docker ps -q -f "name=^${name}$")" ]; then
    echo "$name is already running - bash scripts/deeptex_renloss_21.sh status" >&2; exit 1
  fi
  [ -n "$(docker ps -aq -f "name=^${name}$")" ] && docker rm "$name" >/dev/null
  [ -f "$LOGS/$name.log" ] && mv "$LOGS/$name.log" "$LOGS/$name.prev_$(date +%Y%m%d_%H%M%S).log"
  local L=/workspace/$LOGS/$name
  ( cd docker && $DC run -d -T --name "$name" "$svc" bash -c \
      "{ date +%FT%T; nvidia-smi -L 2>/dev/null; $cmd; } > $L.log 2>&1; echo \"EXIT \$?\" >> $L.log" )
  echo "started $name   log: $LOGS/$name.log   status: bash scripts/deeptex_renloss_21.sh status"
}

need() { [ -e "$1" ] || { echo "missing $1 - $2" >&2; exit 1; }; }

case "${1:-}" in
  grouped)
    need runs/datasets/matsynth/pairs_manifest.json "run: bash scripts/data.sh pairs (and let it finish)"
    detached grouped deeptex "python /workspace/variants/deeptex/render_loss/build/texsr_build_grouped_tfrecord.py \
            --root $PAIRS --out $GROUPED && python /workspace/variants/deeptex/render_loss/build/texsr_verify_grouped.py \
            --tfrecord $GROUPED --root $PAIRS" ;;
  smoke)
    need runs/deeptex/datasets/matsynth_grouped/verify_report.json "run: bash scripts/deeptex_renloss_21.sh grouped"
    detached smoke deeptex "$TRAIN --max-steps 200 --save-steps 100 \
            --checkpoint-root /workspace/runs/deeptex/deeptex_renloss_21_smoke/checkpoints --log-root /workspace/runs/deeptex/deeptex_renloss_21_smoke/logs" ;;
  train)
    need runs/deeptex/datasets/matsynth_grouped/verify_report.json "run: bash scripts/deeptex_renloss_21.sh grouped"
    detached train deeptex "$TRAIN" ;;
  stop)
    docker stop "deeptex_renloss_21_${2:?stage name, e.g. train}" ;;
  status)
    for s in grouped smoke train; do
      n=deeptex_renloss_21_$s
      [ -n "$(docker ps -aq -f "name=^${n}$")" ] || continue
      docker inspect -f "== $n: {{.State.Status}} exit={{.State.ExitCode}} oom={{.State.OOMKilled}} started={{.State.StartedAt}}" "$n"
      [ -f "$LOGS/$n.log" ] && tr '\r' '\n' < "$LOGS/$n.log" | grep -v '^\s*$' | tail -"${LINES:-6}" | cut -c1-200
    done
    [ -f runs/datasets/matsynth/pairs_manifest.json ] && echo "pairs: $(ls runs/datasets/matsynth/_meta | wc -l) materials (scripts/data.sh)"
    for d in runs/deeptex/deeptex_renloss_21_smoke/checkpoints/esrgan_40k runs/deeptex/deeptex_renloss_21/checkpoints/esrgan_40k; do
      [ -d "$d" ] && echo "checkpoints $d: $(ls "$d" | grep -o 'ckpt-[0-9]*' | sort -u | tr '\n' ' ')"
    done
    nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader ;;
  *)
    sed -n '2,19p' "$0"; exit 1 ;;
esac
