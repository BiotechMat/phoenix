#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export DYLD_LIBRARY_PATH="/opt/homebrew/opt/libomp/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"

./scripts/tune_macos_kernel.sh
.venv/bin/pytest tests/test_architecture_verification.py -q
.venv/bin/mypy core sleeves risk sentiment research ui
echo "Starting Control Center on http://127.0.0.1:8080 ..."
exec .venv/bin/uvicorn ui.api_server:app --host 127.0.0.1 --port 8080
