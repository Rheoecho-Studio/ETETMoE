#!/usr/bin/env python3
"""
ETET Step 1: Environment Setup

This script:
1. Verifies the active Python and CUDA environment.
2. Installs the CUDA 12.8 PyTorch build.
3. Installs the core ETET Python dependencies.
4. Installs Streamlit for Web UI.
5. Verifies the installed packages and writes a complete log to output/env_output.log.

The script assumes the user has already activated the intended Python 3.12 Conda environment.
"""

from __future__ import annotations

import importlib
import logging
import os
import platform
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "output"
LOG_FILE = OUTPUT_DIR / "env_output.log"

PYTHON_REQUIRED = (3, 12)
CUDA_TARGET = "cu128"

# Keep the package list compatible with the ETET plan and Hugging Face ecosystem.
CORE_PACKAGES = [
    "modelscope",
    "safetensors",
    "transformers",
    "accelerate",
    "datasets",
]

# Additional packages for Web UI
WEBUI_PACKAGES = [
    "streamlit",
    "textual",
]


def setup_logging() -> logging.Logger:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("ETET-Step1")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


logger = setup_logging()


def run_command(
    command: list[str],
    *,
    check: bool = True,
    capture_output: bool = True,
) -> subprocess.CompletedProcess[str]:
    logger.info("Running: %s", " ".join(command))
    return subprocess.run(
        command,
        check=check,
        text=True,
        capture_output=capture_output,
    )


def log_command_result(result: subprocess.CompletedProcess[str]) -> None:
    if result.stdout:
        logger.info("stdout:\n%s", result.stdout.strip())
    if result.stderr:
        logger.info("stderr:\n%s", result.stderr.strip())


def check_python() -> None:
    version = sys.version_info[:3]
    logger.info("Python version: %s", platform.python_version())
    logger.info("Python executable: %s", sys.executable)

    if version[:2] != PYTHON_REQUIRED:
        raise RuntimeError(
            "ETET requires Python 3.12. "
            f"Detected Python {version[0]}.{version[1]}.{version[2]}."
        )


def check_conda() -> None:
    conda_prefix = os.environ.get("CONDA_PREFIX")
    conda_default_env = os.environ.get("CONDA_DEFAULT_ENV")

    if conda_prefix:
        logger.info("Conda environment: %s", conda_default_env or "<unnamed>")
        logger.info("Conda prefix: %s", conda_prefix)
    else:
        logger.warning(
            "CONDA_PREFIX is not set. The ETET plan expects an activated Conda environment."
        )


def check_nvidia_smi() -> None:
    try:
        result = run_command(["nvidia-smi"], check=False)
        log_command_result(result)

        if result.returncode != 0:
            logger.warning(
                "nvidia-smi returned exit code %d. CUDA verification may fail.",
                result.returncode,
            )
    except FileNotFoundError:
        logger.warning("nvidia-smi was not found in PATH.")


def upgrade_pip_tools() -> None:
    result = run_command(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "pip",
            "setuptools",
            "wheel",
        ]
    )
    log_command_result(result)


def install_pytorch_cuda128() -> None:
    logger.info("Installing PyTorch with CUDA target %s.", CUDA_TARGET)

    result = run_command(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "torch",
            "torchvision",
            "torchaudio",
            "--index-url",
            "https://download.pytorch.org/whl/cu128",
        ]
    )
    log_command_result(result)


def install_core_packages() -> None:
    logger.info("Installing ETET core dependencies.")

    result = run_command(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            *CORE_PACKAGES,
        ]
    )
    log_command_result(result)


def install_streamlit() -> None:
    """Install Streamlit for Web UI."""
    logger.info("Installing Streamlit for Web UI.")

    result = run_command(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            *WEBUI_PACKAGES,
        ]
    )
    log_command_result(result)


def verify_package(package_name: str, import_name: str | None = None) -> None:
    module_name = import_name or package_name

    try:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", "unknown")
        logger.info("OK: %-14s version=%s", package_name, version)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to import {package_name}: {exc}"
        ) from exc


def verify_environment() -> None:
    logger.info("Verifying installed environment.")

    verify_package("torch")
    verify_package("torchvision")
    verify_package("torchaudio")
    verify_package("modelscope")
    verify_package("safetensors")
    verify_package("transformers")
    verify_package("accelerate")
    verify_package("datasets")
    verify_package("streamlit")

    import torch

    logger.info("PyTorch CUDA available: %s", torch.cuda.is_available())
    logger.info("PyTorch compiled CUDA version: %s", torch.version.cuda)

    if torch.cuda.is_available():
        logger.info("CUDA device count: %d", torch.cuda.device_count())

        for index in range(torch.cuda.device_count()):
            logger.info(
                "CUDA device %d: %s",
                index,
                torch.cuda.get_device_name(index),
            )

        logger.info("CUDA device capability: %s", torch.cuda.get_device_capability(0))

        if torch.version.cuda != "12.8":
            logger.warning(
                "PyTorch reports CUDA %s instead of the requested CUDA 12.8 build.",
                torch.version.cuda,
            )
    else:
        logger.warning(
            "PyTorch cannot access CUDA. Check the NVIDIA driver, CUDA compatibility, "
            "and the active Python environment."
        )


def main() -> int:
    logger.info("=" * 72)
    logger.info("ETET Step 1: Environment Setup")
    logger.info("=" * 72)

    try:
        check_python()
        check_conda()
        check_nvidia_smi()

        upgrade_pip_tools()
        install_pytorch_cuda128()
        install_core_packages()
        install_streamlit()  # New: install Streamlit
        verify_environment()

        logger.info("=" * 72)
        logger.info("ETET Step 1 completed successfully.")
        logger.info("Log file: %s", LOG_FILE)
        logger.info("")
        logger.info("To launch the Web UI, run:")
        logger.info("  streamlit run test.py")
        logger.info("=" * 72)
        return 0

    except subprocess.CalledProcessError as exc:
        logger.error(
            "Command failed with exit code %s: %s",
            exc.returncode,
            exc.cmd,
        )
        if exc.stdout:
            logger.error("stdout:\n%s", exc.stdout.strip())
        if exc.stderr:
            logger.error("stderr:\n%s", exc.stderr.strip())
        return exc.returncode or 1

    except Exception as exc:
        logger.exception("ETET Step 1 failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())