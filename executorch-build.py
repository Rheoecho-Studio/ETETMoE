#!/usr/bin/env python3
"""ETET helper: build and install ExecuTorch with XNNPACK and optional Vulkan backend."""
from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "output"
LOG_FILE = OUTPUT_DIR / "executorch_build_output.log"

EXECUTORCH_REPO_URL = "https://github.com/pytorch/executorch.git"
SUPPORTED_CUDA_VERSIONS = {(12, 6), (13, 0), (13, 2), (13, 4)}
NIGHTLY_URL_BASE = "https://download.pytorch.org/whl/nightly"


def setup_logging():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("etet_executorch_build")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setFormatter(formatter)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


LOGGER = setup_logging()


def parse_args():
    parser = argparse.ArgumentParser(description="Build and install ExecuTorch for ETET PTE export.")
    parser.add_argument("--repo-dir", type=Path, default=None,
                        help="ExecuTorch source directory (default: ~/executorch).")
    parser.add_argument("--branch", type=str, default=None, help="Branch or tag to checkout after cloning.")
    parser.add_argument("--no-vulkan", action="store_true", help="Skip the Vulkan backend (XNNPACK only).")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU-only PyTorch nightly instead of CUDA wheels.")
    parser.add_argument("--no-submodules", action="store_true", help="Skip submodule sync/update.")
    parser.add_argument("--minimal", "-m", action="store_true",
                        help="Pass --minimal to install_executorch.py (skip example-only packages).")
    parser.add_argument("--verbose", "-v", action="store_true", help="Pass --verbose to install_executorch.py.")
    parser.add_argument("--check-only", action="store_true", help="Only verify an existing installation.")
    parser.add_argument("--assume-yes", "-y", action="store_true", help="Skip interactive confirmations.")
    return parser.parse_args()


def confirm(prompt: str, assume_yes: bool) -> bool:
    if assume_yes:
        return True
    answer = input(prompt).strip().lower()
    return answer in ("y", "yes")


def run(cmd, cwd=None, env=None, check=True) -> int:
    # Stream child output into the log so long builds stay observable.
    LOGGER.info("Running: %s", " ".join(cmd))
    process = subprocess.Popen(cmd, cwd=cwd, env=env, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if process.stdout is not None:
        for line in process.stdout:
            LOGGER.info(line.rstrip())
    code = process.wait()
    if check and code != 0:
        raise RuntimeError(f"Command failed ({code}): {' '.join(cmd)}")
    return code


def installed_torch_version():
    result = subprocess.run([sys.executable, "-c", "import torch; print(torch.__version__)"],
                            capture_output=True, text=True)
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def detect_cuda_version():
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        LOGGER.info("nvcc not found in PATH; the CPU PyTorch index will be used.")
        return None
    result = subprocess.run([nvcc, "--version"], capture_output=True, text=True)
    match = re.search(r"release (\d+)\.(\d+)", result.stdout)
    if not match:
        LOGGER.info("Could not parse nvcc version; the CPU PyTorch index will be used.")
        return None
    version = (int(match.group(1)), int(match.group(2)))
    LOGGER.info("Detected CUDA toolkit: %d.%d", version[0], version[1])
    return version


def torch_index(args):
    if args.cpu:
        LOGGER.info("CPU-only mode requested.")
        return f"{NIGHTLY_URL_BASE}/cpu"
    version = detect_cuda_version()
    if version is None:
        return f"{NIGHTLY_URL_BASE}/cpu"
    if version not in SUPPORTED_CUDA_VERSIONS:
        LOGGER.warning("CUDA %d.%d is not supported by ExecuTorch %s; falling back to CPU wheels.",
                       version[0], version[1], sorted(SUPPORTED_CUDA_VERSIONS))
        return f"{NIGHTLY_URL_BASE}/cpu"
    return f"{NIGHTLY_URL_BASE}/cu{version[0]}{version[1]}"


def glslc_available() -> bool:
    if shutil.which("glslc"):
        return True
    sdk = os.environ.get("VULKAN_SDK")
    if sdk:
        candidate = Path(sdk) / "bin" / "glslc"
        return candidate.exists()
    return False


def environment_label() -> str:
    if os.environ.get("CONDA_DEFAULT_ENV"):
        return f"conda:{os.environ['CONDA_DEFAULT_ENV']}"
    if os.environ.get("VIRTUAL_ENV"):
        return f"venv:{os.environ['VIRTUAL_ENV']}"
    return "system"


def preflight(args) -> None:
    torch_version = installed_torch_version()
    LOGGER.info("Python: %s | interpreter: %s | environment: %s",
                sys.version.split()[0], sys.executable, environment_label())
    LOGGER.info("Installed torch before build: %s", torch_version or "<not installed>")

    if torch_version and torch_version.startswith("2.11"):
        LOGGER.warning("The active interpreter has torch %s (looks like an ETET training environment).", torch_version)
        LOGGER.warning("The build will replace torch with a nightly build and may break train.py / visiontrain.py.")
        if not confirm("Continue anyway? [y/N]: ", args.assume_yes):
            raise SystemExit("Aborted. Activate a dedicated environment (conda, venv or uv) and rerun.")

    if shutil.which("git") is None:
        raise RuntimeError("git is required but was not found in PATH.")

    inside_project = PROJECT_ROOT / "executorch"
    if inside_project.exists():
        # executorch is a namespace package: this directory shadows the installed one whenever
        # python runs from the ETET project, which breaks imports at runtime.
        LOGGER.error("Found %s inside the ETET project.", inside_project)
        LOGGER.error("It shadows the installed executorch package and breaks imports from this directory.")
        raise RuntimeError(f"Delete it first: rm -rf {inside_project}")


def prepare_repo(args) -> Path:
    repo_dir = args.repo_dir or (Path.home() / "executorch")
    if PROJECT_ROOT in repo_dir.parents:
        LOGGER.warning("Repo dir %s is inside the ETET project; this pollutes the repository.", repo_dir)

    install_script = repo_dir / "install_executorch.py"
    if install_script.exists():
        LOGGER.info("Using existing ExecuTorch source: %s", repo_dir)
    else:
        LOGGER.info("Cloning ExecuTorch into %s (this takes a while).", repo_dir)
        run(["git", "clone", EXECUTORCH_REPO_URL, str(repo_dir)])
        if not install_script.exists():
            raise RuntimeError(f"Clone finished but {install_script} is missing.")

    if args.branch:
        run(["git", "-C", str(repo_dir), "checkout", args.branch])
    LOGGER.info("Repo ready: %s", repo_dir)
    return repo_dir


def update_submodules(repo_dir: Path) -> None:
    # Nested submodules under extension/llm/tokenizers are required by pytorch_tokenizers.
    LOGGER.info("Syncing git submodules (recursive, several hundred MB).")
    run(["git", "-C", str(repo_dir), "submodule", "sync", "--recursive"])
    run(["git", "-C", str(repo_dir), "submodule", "update", "--init", "--recursive"])
    LOGGER.info("Submodules updated.")


def install_cpu_packages(repo_dir: Path, args, index_url: str, env) -> None:
    # The official installer pins torchao 0.18.0.dev on the CPU branch, which conflicts with
    # executorch's torchao>=0.19.0.dev requirement, so install nightly packages explicitly.
    LOGGER.info("CPU mode: installing nightly torch and torchao from %s", index_url)
    run([sys.executable, "-m", "pip", "install", "-r", "requirements-dev.txt",
         "--extra-index-url", index_url], cwd=str(repo_dir), env=env)
    run([sys.executable, "-m", "pip", "install", "--pre", "torch",
         "--index-url", index_url], cwd=str(repo_dir), env=env)
    run([sys.executable, "-m", "pip", "install", "--pre", "torchao",
         "--extra-index-url", index_url], cwd=str(repo_dir), env=env)
    LOGGER.info("Installing the ExecuTorch package (this takes a while).")
    run([sys.executable, "-m", "pip", "install", ".", "--no-build-isolation",
         "--extra-index-url", index_url], cwd=str(repo_dir), env=env)
    if not args.minimal:
        # Optional: only needed by executorch example runners, harmless if it fails.
        LOGGER.info("Installing the bundled pytorch_tokenizers package (optional).")
        run([sys.executable, "-m", "pip", "install", "--no-build-isolation",
             "extension/llm/tokenizers", "--extra-index-url", index_url],
            cwd=str(repo_dir), env=env, check=False)


def build_and_install(repo_dir: Path, args, index_url: str) -> None:
    env = os.environ.copy()
    if not args.no_vulkan:
        if not glslc_available():
            LOGGER.warning("glslc not found; Vulkan backend cannot be built.")
            if confirm("Fall back to XNNPACK-only build? [y/N]: ", args.assume_yes):
                args.no_vulkan = True
            else:
                raise RuntimeError("Install glslc (shaderc or the LunarG Vulkan SDK) and rerun.")
        else:
            env["CMAKE_ARGS"] = "-DEXECUTORCH_BUILD_VULKAN=ON"

    LOGGER.info("PyTorch index used: %s", index_url)
    if args.cpu:
        install_cpu_packages(repo_dir, args, index_url, env)
    else:
        LOGGER.info("Installing dependencies and the ExecuTorch package (this takes a while).")
        command = [sys.executable, "install_executorch.py"]
        if args.minimal:
            command.append("--minimal")
        if args.verbose:
            command.append("--verbose")
        run(command, cwd=str(repo_dir), env=env)
    LOGGER.info("ExecuTorch installation finished.")


BACKEND_IMPORTS = {
    "xnnpack": ("executorch.backends.xnnpack.partition.xnnpack_partitioner", "XnnpackPartitioner"),
    "vulkan": ("executorch.backends.vulkan.partitioner.vulkan_partitioner", "VulkanPartitioner"),
}


def backend_library_path(backend: str):
    # The partitioner is pure Python, so importing it proves nothing about the native backend.
    import importlib.util
    spec = importlib.util.find_spec("executorch")
    if spec is None:
        return None
    # executorch is a namespace package: origin is None and several search locations can shadow
    # each other, so every location must be scanned.
    locations = list(spec.submodule_search_locations or [])
    if spec.origin:
        locations.append(str(Path(spec.origin).parent))
    for location in locations:
        for path in Path(location).rglob("*.so"):
            if f"backend_{backend}" in path.name:
                return str(path)
    return None


def lowering_smoke(backend: str) -> bool:
    module_name, class_name = BACKEND_IMPORTS[backend]
    snippet = (
        "import torch, torch.nn as nn\n"
        "from torch.export import export\n"
        "from executorch.exir import to_edge_transform_and_lower\n"
        f"from {module_name} import {class_name}\n"
        "class Tiny(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.linear = nn.Linear(8, 8, bias=False)\n"
        "    def forward(self, x):\n"
        "        return self.linear(x)\n"
        "ep = export(Tiny().eval(), (torch.randn(1, 8),))\n"
        f"et = to_edge_transform_and_lower(ep, partitioner=[{class_name}()]).to_executorch()\n"
        "assert len(et.buffer) > 0\n"
        "print('lowering OK')\n"
    )
    # Run outside the project so a stale executorch directory cannot shadow the installed package.
    with tempfile.TemporaryDirectory() as tmp_dir:
        result = subprocess.run([sys.executable, "-c", snippet], capture_output=True, text=True, cwd=tmp_dir)
    if result.returncode == 0:
        LOGGER.info("%s lowering smoke: OK", backend)
        return True
    LOGGER.error("%s lowering smoke failed:", backend)
    for line in (result.stderr or "").strip().splitlines()[-8:]:
        LOGGER.error(line)
    return False


def verify(args) -> None:
    failures = []
    basic_checks = [
        "import executorch; print('executorch OK')",
        "from executorch.exir import to_edge_transform_and_lower; print('to_edge OK')",
    ]
    for snippet in basic_checks:
        result = subprocess.run([sys.executable, "-c", snippet], capture_output=True, text=True)
        if result.returncode == 0:
            LOGGER.info("Verified: %s", result.stdout.strip().splitlines()[-1])
        else:
            LOGGER.error("Verification failed: %s", snippet)
            failures.append(snippet)

    backends = ["xnnpack"] if args.no_vulkan else ["xnnpack", "vulkan"]
    for backend in backends:
        library = backend_library_path(backend)
        if library:
            LOGGER.info("%s native backend library: %s", backend, library)
        else:
            # Some backends (Vulkan) are static archives linked into the pybind extension, so a
            # missing standalone .so proves nothing; the lowering smoke below is decisive.
            LOGGER.info("%s has no standalone .so (likely statically linked); using lowering smoke.", backend)
        if not lowering_smoke(backend):
            failures.append(f"lowering:{backend}")

    if failures:
        if "lowering:vulkan" in failures:
            LOGGER.error("Vulkan lowering failed: the native backend is not usable.")
            LOGGER.error("Make sure glslc is in PATH and rerun this script to rebuild with Vulkan enabled.")
        raise RuntimeError(f"{len(failures)} verification check(s) failed; see {LOG_FILE}")


def print_next_steps(args) -> None:
    LOGGER.info("=" * 72)
    LOGGER.info("Next steps")
    LOGGER.info("  1. Back in the ETET project, run the P0 export:")
    LOGGER.info("     python pteoutput.py --model models/ETET_VL_Preview --target fp16-xnnpack --ctx 2048")
    if not args.no_vulkan:
        LOGGER.info("  2. Once P0 passes, add the Vulkan target:")
        LOGGER.info("     python pteoutput.py --model models/ETET_VL_Preview --target fp16-vulkan --ctx 2048")
    LOGGER.info("Log file: %s", LOG_FILE)
    LOGGER.info("=" * 72)


def main() -> int:
    args = parse_args()
    LOGGER.info("=" * 72)
    LOGGER.info("ETET: ExecuTorch build helper")
    LOGGER.info("Manual prerequisite: activate a dedicated environment first (conda, venv or uv, Python 3.12).")
    LOGGER.info("  Example: python -m venv ~/etet-pte && source ~/etet-pte/bin/activate")
    LOGGER.info("=" * 72)

    preflight(args)
    index_url = torch_index(args)

    if args.check_only:
        verify(args)
        LOGGER.info("Existing installation verified.")
        return 0

    repo_dir = prepare_repo(args)
    if not args.no_submodules:
        update_submodules(repo_dir)
    build_and_install(repo_dir, args, index_url)
    verify(args)
    print_next_steps(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        LOGGER.exception("ExecuTorch build failed: %s", exc)
        raise SystemExit(1)
