#!/usr/bin/env bash
# ==============================================================================
# Apple MacBook Pro M1 Max (64GB Unified Memory) Non-Interactive Kernel Tuning
# ==============================================================================
# Sets the runtime Metal/IOGPU wired memory ceiling to 40,960 MB (40.0 GB),
# leaving 24.0 GB guaranteed for the macOS kernel, network stack, Redis/Arrow
# IPC, and NautilusTrader Rust Engine.
#
# SAFETY MANDATES:
# 1. Uses non-interactive `sudo -n` so automated builds/tests never hang waiting
#    for a password prompt.
# 2. Never modifies `/etc/sysctl.conf` to preserve reboot safety.
# ==============================================================================

set -u

WIRED_LIMIT_MB=40960

echo "[hardware_bootstrap] Attempting non-interactive iogpu.wired_limit_mb=${WIRED_LIMIT_MB}..."

if sudo -n sysctl "iogpu.wired_limit_mb=${WIRED_LIMIT_MB}" >/dev/null 2>&1; then
    echo "[hardware_bootstrap] SUCCESS: iogpu.wired_limit_mb set to ${WIRED_LIMIT_MB} MB (40.0 GB)."
else
    echo "[hardware_bootstrap] WARNING: Non-interactive sudo -n privilege not cached; continuing gracefully with default macOS Metal working set limit."
fi

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export POLARS_MAX_THREADS=4
export PYTORCH_MPS_LOW_WATERMARK_RATIO="0.5"
export PYTORCH_MPS_HIGH_WATERMARK_RATIO="0.7"
export PYTORCH_MPS_FAST_MATH="1"

exit 0
