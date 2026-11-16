#!/usr/bin/env bash
# Where a run started by run_example.sh stands. Safe to run any time, from any SSH
# session; it only reads.
#
#   bash variants/pbr_sr/server/status.sh [RUN_NAME]     (default cradle_globe_paper)
#   LINES=40 bash variants/pbr_sr/server/status.sh RUN   (longer log tail)
set -uo pipefail
export MSYS_NO_PATHCONV=1

REPO=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$REPO"
RUN=${1:-cradle_globe_paper}
NAME=pbrsr_$RUN
LOGS=runs/pbr_sr/logs
VIEWS=750   # pseudo-GT views of the README example (circle_n_view x elevations)

echo "== $RUN  ($(date +%FT%T))"
[ -f "$LOGS/$RUN.start" ] && head -1 "$LOGS/$RUN.start"
if [ -n "$(docker ps -aq -f "name=^${NAME}$")" ]; then
  docker inspect -f 'container: {{.State.Status}}  exit={{.State.ExitCode}}  oom_killed={{.State.OOMKilled}}  started={{.State.StartedAt}}' "$NAME"
else
  echo "container: none"
fi
[ -f "$LOGS/$RUN.end" ] && echo "ended: $(cat "$LOGS/$RUN.end")  $(grep '^EXIT' "$LOGS/$RUN.log" | tail -1)"

d=$(ls -d runs/pbr_sr/$RUN/cache_render/*/ 2>/dev/null | head -1)
if [ -n "$d" ]; then
  n=$(ls "$d" | grep -c _HQ_image)
  echo "pseudo-GT cache: $n / $VIEWS views"
  if [ "$n" -ge 2 ] && [ "$n" -lt "$VIEWS" ]; then
    first=$(stat -c %Y "$(ls -tr "$d"*_HQ_image.png | head -1)")
    last=$(stat -c %Y "$(ls -t "$d"*_HQ_image.png | head -1)")
    per=$(( (last - first) / (n - 1) ))
    echo "  ${per} s/view, cache ETA $(date -d "@$(( last + per * (VIEWS - n) ))" +'%m-%d %H:%M')"
  fi
fi
ls runs/pbr_sr/$RUN/cache_render/*.pt >/dev/null 2>&1 && echo "cache .pt written (optimization reached)"

if [ -f "$LOGS/$RUN.log" ]; then
  echo "== log tail ($LOGS/$RUN.log)"
  tr '\r' '\n' < "$LOGS/$RUN.log" | grep -v 'Setting up\|building SDP\|^\s*$\|B/s\]$\|Tiled VAE\|Tiled Processing\|Executing' \
    | tail -"${LINES:-12}" | cut -c1-200
  grep -nE 'Traceback|\(Pdb\)|BdbQuit|CUDA out of memory|Error' "$LOGS/$RUN.log" | tail -3
fi

echo "== GPU"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu --format=csv,noheader
echo "== RAM (GB)"
if command -v free >/dev/null; then free -g | head -2; else awk '/MemTotal|MemAvailable/' /proc/meminfo 2>/dev/null; fi
