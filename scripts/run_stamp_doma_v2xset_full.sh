#!/usr/bin/env bash
# Run in an existing shell/screen; no remote execution or session creation.
set -euo pipefail
STAMP_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$STAMP_ROOT"
export PYTHONPATH="$STAMP_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
exec "${PYTHON:-python}" -B opencood/tools/run_stamp_doma_v2xset.py "$@"
