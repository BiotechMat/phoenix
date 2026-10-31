"""Hardware topology, 64GB M1 Max memory partitioning, thread pinning, and macOS spawn bootstrap."""

from __future__ import annotations

import json
import logging
import multiprocessing
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from core.contracts import EnvironmentConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BootstrapStatus:
    """Result of executing the M1 Max 64GB hardware bootstrap."""

    start_method: str
    wired_limit_mb: int
    sysctl_applied: bool
    sysctl_message: str
    omp_threads: int
    polars_threads: int
    active_memory_budget_gb: float
    free_headroom_gb: float


def enforce_spawn_and_thread_caps(omp_threads: int = 4, polars_threads: int = 4) -> str:
    """Enforce macOS 'spawn' multiprocessing and cap OpenMP/MKL/OpenBLAS/Polars threads.

    Must be called at the top of every worker entrypoint before importing numerical libraries.
    """
    multiprocessing.set_start_method("spawn", force=True)
    str_omp = str(omp_threads)
    str_polars = str(polars_threads)

    os.environ["OMP_NUM_THREADS"] = str_omp
    os.environ["MKL_NUM_THREADS"] = str_omp
    os.environ["OPENBLAS_NUM_THREADS"] = str_omp
    os.environ["POLARS_MAX_THREADS"] = str_polars

    # PyTorch / Metal MPS auxiliary pool watermarks (10.0 GB max pool)
    os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = "0.5"
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.7"
    os.environ["PYTORCH_MPS_FAST_MATH"] = "1"

    current_method = multiprocessing.get_start_method(allow_none=True) or "spawn"
    return current_method


def configure_macos_wired_memory(limit_mb: int = 40960) -> tuple[bool, str]:
    """Non-interactively set iogpu.wired_limit_mb via sudo -n sysctl with 2s timeout.

    Never prompts for an interactive password and never writes to /etc/sysctl.conf
    to preserve reboot safety and prevent hanging automated tests.
    """
    if limit_mb > 40960:
        raise ValueError(f"Requested wired_limit_mb={limit_mb} exceeds 40,960 MB (40.0 GB) M1 Max ceiling.")

    cmd = ["sudo", "-n", "sysctl", f"iogpu.wired_limit_mb={limit_mb}"]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
        if proc.returncode == 0:
            msg = f"Applied iogpu.wired_limit_mb={limit_mb} via non-interactive sysctl."
            logger.info(msg)
            return True, msg

        stderr_clean = (proc.stderr or proc.stdout or "sudo -n privilege not cached").strip()
        warn_msg = (
            f"Non-interactive sysctl skipped (code={proc.returncode}: {stderr_clean}); "
            "continuing gracefully with default macOS Metal working set limit."
        )
        logger.warning(warn_msg)
        return False, warn_msg
    except (subprocess.TimeoutExpired, FileNotFoundError, PermissionError, OSError) as exc:
        warn_msg = (
            f"Non-interactive sysctl unavailable ({type(exc).__name__}: {exc}); "
            "continuing gracefully with default macOS Metal working set limit."
        )
        logger.warning(warn_msg)
        return False, warn_msg


def bootstrap_m1max_hardware(config: EnvironmentConfig | None = None) -> BootstrapStatus:
    """Execute full 64GB M1 Max hardware bootstrap (spawn isolation, thread caps, wired limit)."""
    cfg = config or load_environment_config("ci_mock")
    start_method = enforce_spawn_and_thread_caps(
        omp_threads=cfg.omp_num_threads,
        polars_threads=cfg.polars_max_threads,
    )
    if cfg.is_mock:
        sysctl_ok, sysctl_msg = False, "Mock profile active; skipping kernel sysctl call."
    else:
        sysctl_ok, sysctl_msg = configure_macos_wired_memory(limit_mb=cfg.wired_limit_mb)

    active_gb = (
        cfg.os_ipc_budget_gb
        + cfg.nautilus_rust_budget_gb
        + cfg.lgbm_polars_budget_gb
        + cfg.pytorch_mps_budget_gb
        + cfg.mlx_llm_budget_gb
    )
    free_gb = 64.0 - active_gb

    return BootstrapStatus(
        start_method=start_method,
        wired_limit_mb=cfg.wired_limit_mb,
        sysctl_applied=sysctl_ok,
        sysctl_message=sysctl_msg,
        omp_threads=int(os.environ["OMP_NUM_THREADS"]),
        polars_threads=int(os.environ["POLARS_MAX_THREADS"]),
        active_memory_budget_gb=active_gb,
        free_headroom_gb=free_gb,
    )


def load_environment_config(
    profile: str = "ci_mock", config_dir: Path | None = None
) -> EnvironmentConfig:
    """Load either prod_m1max_64gb or ci_mock single-line JSON profile from config/."""
    base_dir = config_dir or Path(__file__).resolve().parent.parent / "config"
    filename = (
        "config_m1max_64gb.json"
        if profile in ("prod_m1max_64gb", "prod", "m1max")
        else "config_ci_mock.json"
    )
    config_path = base_dir / filename
    raw_text = config_path.read_text(encoding="utf-8").strip()
    data = json.loads(raw_text)
    return EnvironmentConfig.model_validate(data)
