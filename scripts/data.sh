#!/usr/bin/env bash
# MatSynth -> the data every arm shares. No model code; the dataprep container only (no GPU).
# Run from the repo root in Git Bash (or any Linux shell):
#
#   bash scripts/data.sh evalset    12 evaluation materials from test-00000 -> runs/matsynth/ (~1-2 min)
#   bash scripts/data.sh check      pair degradation == evaluation degradation, pixel for pixel ([PASS])
#   bash scripts/data.sh pairs      all train shards -> 512/128 PNG pairs -> runs/datasets/matsynth/
#                                   (detached, ~1 h from an HDD; rerun resumes per material)
#   bash scripts/data.sh status     container, log tail, materials done
#   bash scripts/data.sh stop       stop pairs
#
# Who reads what:
#   runs/matsynth/              evaluation set (LR input + 512 GT); excluded from the pairs
#   runs/datasets/matsynth/     <map>/{HR,LRx4}/<material>__<map>.png, _meta/, pairs_manifest.json
#                               -> scripts/deeptex_renloss_21.sh grouped (deeptex_renloss_21's TFRecord), scripts/matup_ft.sh
# Code: variants/util/matsynth_extract.py, variants/deeptex/matsynth/matsynth_make_pairs.py
set -euo pipefail
export MSYS_NO_PATHCONV=1

REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
LOGS=runs/logs
mkdir -p "$LOGS"
if docker compose version >/dev/null 2>&1; then DC="docker compose"; else DC="docker-compose"; fi
PAIRS=runs/datasets/matsynth
NAME=data_pairs

need() { [ -e "$1" ] || { echo "missing $1 - $2" >&2; exit 1; }; }

case "${1:-}" in
  evalset)
    ( cd docker && $DC run --rm -T dataprep python variants/util/matsynth_extract.py --index 0 --count 12 ) ;;
  check)
    need runs/matsynth "run: bash scripts/data.sh evalset"
    ( cd docker && $DC run --rm -T dataprep python variants/deeptex/matsynth/matsynth_make_pairs.py --check-eval ) ;;
  pairs)
    need runs/matsynth "run: bash scripts/data.sh evalset"
    if [ -n "$(docker ps -q -f "name=^${NAME}$")" ]; then echo "$NAME is already running - bash scripts/data.sh status" >&2; exit 1; fi
    [ -n "$(docker ps -aq -f "name=^${NAME}$")" ] && docker rm "$NAME" >/dev/null
    [ -f "$LOGS/$NAME.log" ] && mv "$LOGS/$NAME.log" "$LOGS/$NAME.prev_$(date +%Y%m%d_%H%M%S).log"
    L=/workspace/$LOGS/$NAME
    ( cd docker && $DC run -d -T --name "$NAME" dataprep bash -c \
        "{ date +%FT%T; python variants/deeptex/matsynth/matsynth_make_pairs.py --out /workspace/$PAIRS --jobs 4; } \
 > $L.log 2>&1; echo \"EXIT \$?\" >> $L.log" )
    echo "started $NAME   log: $LOGS/$NAME.log   status: bash scripts/data.sh status" ;;
  stop)
    docker stop "$NAME" ;;
  status)
    [ -d runs/matsynth ] && echo "evalset: $(ls runs/matsynth | wc -l) materials in runs/matsynth" || echo "evalset: missing"
    if [ -n "$(docker ps -aq -f "name=^${NAME}$")" ]; then
      docker inspect -f "== $NAME: {{.State.Status}} exit={{.State.ExitCode}} oom={{.State.OOMKilled}} started={{.State.StartedAt}}" "$NAME"
      [ -f "$LOGS/$NAME.log" ] && tr '\r' '\n' < "$LOGS/$NAME.log" | grep -v '^\s*$' | tail -"${LINES:-6}" | cut -c1-200
    fi
    [ -d $PAIRS/_meta ] && echo "pairs: $(ls $PAIRS/_meta | wc -l) materials done in $PAIRS"
    [ -f $PAIRS/pairs_manifest.json ] && echo "pairs_manifest.json present (a pairs run finished)"
    true ;;
  *)
    sed -n '2,17p' "$0"; exit 1 ;;
esac
