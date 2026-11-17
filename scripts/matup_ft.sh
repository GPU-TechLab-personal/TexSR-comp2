#!/usr/bin/env bash
# MatUp's SwinIR-L x4 fine-tuned on MatSynth (comp1's better SwinIR recipe: fixed area
# pairs, L1, KAIR's official trainer). Uses the pairs of scripts/r1.sh pairs.
# Run from the repo root in Git Bash:
#
#   bash scripts/matup_ft.sh train       40k steps -> runs/matup_ft/kair/swinirL_ft/models/<N>_E.pth
#                                        (detached; rerun resumes from the last saved step)
#   bash scripts/matup_ft.sh smoke       20 steps into runs/matup_ft_smoke/ (detached, ~1 min)
#   bash scripts/matup_ft.sh status      container, log tail, saved steps, GPU
#   bash scripts/matup_ft.sh stop        stop training
#
# Then MatUp with the fine-tuned weights on Cradle_Globe:
#   bash scripts/init.sh matup_ft 40000 && bash scripts/init.sh pack matup_ft
#   INIT=matup_ft bash variants/pbr_sr/server/run_example.sh
# Recipe and reasons: variants/matup_ft/swinirL_ft.json, variants/matup_ft/train.py.
set -euo pipefail
export MSYS_NO_PATHCONV=1

REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
mkdir -p runs/logs
if docker compose version >/dev/null 2>&1; then DC="docker compose"; else DC="docker-compose"; fi
W=/workspace
MODELS=runs/matup_ft/kair/swinirL_ft/models

detached() {  # name, command
  local name=$1 cmd=$2
  if [ -n "$(docker ps -q -f "name=^${name}$")" ]; then echo "$name is already running" >&2; exit 1; fi
  [ -n "$(docker ps -aq -f "name=^${name}$")" ] && docker rm "$name" >/dev/null
  [ -f runs/logs/$name.log ] && mv runs/logs/$name.log runs/logs/$name.prev_$(date +%Y%m%d_%H%M%S).log
  ( cd docker && $DC run -d -T --name "$name" matup bash -c \
      "{ date +%FT%T; nvidia-smi -L; $cmd; } > $W/runs/logs/$name.log 2>&1; echo \"EXIT \$?\" >> $W/runs/logs/$name.log" )
  echo "started $name   log: runs/logs/$name.log   status: bash scripts/matup_ft.sh status"
}

case "${1:-}" in
  train)
    [ -d runs/deeptex/datasets/matsynth/_meta ] || { echo "no MatSynth pairs - run: bash scripts/r1.sh pairs" >&2; exit 1; }
    detached matup_ft "python $W/variants/matup_ft/train.py" ;;
  smoke)
    [ -d runs/deeptex/datasets/matsynth/_meta ] || { echo "no MatSynth pairs - run: bash scripts/r1.sh pairs" >&2; exit 1; }
    detached matup_ft_smoke "python $W/variants/matup_ft/train.py --total-iter 20 \
 --set train.checkpoint_save=10 --set train.checkpoint_print=5 --set path.root=$W/runs/matup_ft_smoke/kair" ;;
  stop)
    docker stop matup_ft ;;
  status)
    for n in matup_ft_smoke matup_ft; do
      [ -n "$(docker ps -aq -f "name=^${n}$")" ] || continue
      docker inspect -f "== $n: {{.State.Status}} exit={{.State.ExitCode}} oom={{.State.OOMKilled}} started={{.State.StartedAt}}" "$n"
      [ -f runs/logs/$n.log ] && grep -E 'iter:|matup_ft\]|Error|Traceback|EXIT' runs/logs/$n.log | tail -"${LINES:-4}" | cut -c1-160
    done
    [ -d $MODELS ] && echo "saved steps: $(ls $MODELS | grep -o '^[0-9]*_E' | sed 's/_E//' | sort -n | tr '\n' ' ')"
    nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader ;;
  *)
    sed -n '2,16p' "$0"; exit 1 ;;
esac
