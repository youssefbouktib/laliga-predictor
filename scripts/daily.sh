#!/usr/bin/env sh
# Daily update for macOS/Linux. Example crontab entry (09:00 every day):
#   0 9 * * * /path/to/laliga-predictor/scripts/daily.sh
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python; [ -x "$PY" ] || PY=python3
mkdir -p reports
{ echo "===== $(date -Iseconds) ====="; $PY run.py download; $PY run.py predict; $PY run.py settle; } >> reports/daily.log 2>&1
