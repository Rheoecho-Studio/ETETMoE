#!/usr/bin/env python3
from __future__ import annotations
import hashlib
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
DATASETS_DIR = PROJECT_ROOT / "datasets"
OUTPUT_DIR = PROJECT_ROOT / "output"
LOG_FILE = OUTPUT_DIR / "download_output.log"

HF_MIRROR = "https://hf-mirror.com"

MINICPM_REPO = "OpenBMB/MiniCPM5-1B-SFT"
SIGLIP_REPO = "LiheYoung/SigLIP-HD"
AYA_DATASET_REPO = "CohereLabs/aya_dataset"
AYA_COLLECTION_REPO = "CohereLabs/aya_collection_language_split"
LLAVA_PRETRAIN_REPO = "liuhaotian/LLaVA-Pretrain"

MODEL_WEIGHT_EXTENSIONS = {
    ".bin",
    ".safetensors",
    ".pt",
    ".pth",
}

DATASET_FILE_EXTENSIONS = {
    ".json",
    ".jsonl",
    ".parquet",
    ".arrow",
    ".csv",
    ".tsv",
    ".txt",
    ".jsonl.zst",
    ".json.zst",
}

def setup_logging() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

def run_command(
    command: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    logging.info("Running: %s", " ".join(command))
    process_env = os.environ.copy()
    if env is not None:
        process_env.update(env)
    result = subprocess.run(command, cwd=cwd, env=process_env, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}: {' '.join(command)}"
        )

def command_exists(command: str) -> bool:
    return shutil.which(command) is not None

def find_hfd() -> Path:
    candidates = [
        PROJECT_ROOT / "hfd.sh",
        PROJECT_ROOT / "scripts" / "hfd.sh",
        PROJECT_ROOT / "tools" / "hfd.sh",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    hfd_path = shutil.which("hfd.sh")
    if hfd_path:
        return Path(hfd_path).resolve()
    raise FileNotFoundError(
        "hfd.sh was not found. "
        "Please place hfd.sh in the project root or add it to PATH."
    )

def directory_has_files(directory: Path) -> bool:
    if not directory.is_dir():
        return False
    for file in directory.rglob("*"):
        if file.is_file():
            return True
    return False

def find_incomplete_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    incomplete_suffixes = (".incomplete", ".part", ".aria2")
    incomplete_names = {".incomplete", ".part", ".aria2"}
    result: list[Path] = []
    for file in directory.rglob("*"):
        if not file.is_file():
            continue
        if file.name in incomplete_names:
            result.append(file)
            continue
        if file.name.endswith(incomplete_suffixes):
            result.append(file)
    return result

def prepare_directories() -> None:
    for directory in (MODELS_DIR, DATASETS_DIR, OUTPUT_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    logging.info("Directory structure initialized.")

def download_modelscope_model(repo_id: str, destination: Path) -> None:
    if directory_has_files(destination):
        logging.info("Skipping existing ModelScope model: %s", destination)
        return
    if not command_exists("modelscope"):
        raise RuntimeError("ModelScope CLI was not found. Please run env.py first.")
    destination.mkdir(parents=True, exist_ok=True)
    logging.info("Downloading ModelScope model: %s", repo_id)
    run_command([
        "modelscope", "download",
        "--model", repo_id,
        "--local_dir", str(destination),
    ])

def download_modelscope_dataset(repo_id: str, destination: Path) -> None:
    if directory_has_files(destination):
        logging.info("Skipping existing ModelScope dataset: %s", destination)
        return
    if not command_exists("modelscope"):
        raise RuntimeError("ModelScope CLI was not found. Please run env.py first.")
    destination.mkdir(parents=True, exist_ok=True)
    logging.info("Downloading ModelScope dataset: %s", repo_id)
    run_command([
        "modelscope", "download",
        "--dataset", repo_id,
        "--local_dir", str(destination),
    ])

def download_huggingface(
    repo_id: str,
    destination: Path,
    hfd_path: Path,
    dataset: bool = False,
    include_patterns: list[str] | None = None,
    force: bool = False,
) -> None:
    if force and destination.exists():
        logging.info("Force download enabled: removing existing directory %s", destination)
        shutil.rmtree(destination)
    if directory_has_files(destination):
        logging.info("Skipping existing Hugging Face resource: %s", destination)
        return
    destination.mkdir(parents=True, exist_ok=True)
    logging.info("Downloading from hf-mirror: %s", repo_id)
    environment = {"HF_ENDPOINT": HF_MIRROR}
    command = [str(hfd_path), repo_id, "--local-dir", str(destination)]
    if dataset:
        command.append("--dataset")
    if include_patterns:
        for pattern in include_patterns:
            command.extend(["--include", pattern])
            logging.info("Adding include pattern: %s", pattern)
    run_command(command, env=environment)

def has_model_weights(directory: Path) -> bool:
    for file in directory.rglob("*"):
        if not file.is_file():
            continue
        if file.suffix.lower() in MODEL_WEIGHT_EXTENSIONS:
            return True
    return False

def has_dataset_files(directory: Path) -> bool:
    for file in directory.rglob("*"):
        if not file.is_file():
            continue
        filename = file.name.lower()
        for extension in DATASET_FILE_EXTENSIONS:
            if filename.endswith(extension):
                return True
    return False

def verify_model_directory(name: str, directory: Path) -> bool:
    if not directory.is_dir():
        logging.error("%s directory does not exist: %s", name, directory)
        return False
    if not directory_has_files(directory):
        logging.error("%s directory is empty: %s", name, directory)
        return False
    incomplete_files = find_incomplete_files(directory)
    if incomplete_files:
        logging.error("%s contains incomplete download artifacts:", name)
        for file in incomplete_files:
            logging.error("  %s", file)
        return False
    config_file = directory / "config.json"
    if not config_file.is_file():
        logging.error("%s is missing config.json.", name)
        return False
    if config_file.stat().st_size == 0:
        logging.error("%s config.json is empty.", name)
        return False
    if not has_model_weights(directory):
        logging.error("%s does not contain recognizable model weight files.", name)
        return False
    logging.info("%s model integrity check passed.", name)
    return True

def verify_dataset_directory(name: str, directory: Path) -> bool:
    if not directory.is_dir():
        logging.error("%s directory does not exist: %s", name, directory)
        return False
    if not directory_has_files(directory):
        logging.error("%s directory is empty: %s", name, directory)
        return False
    incomplete_files = find_incomplete_files(directory)
    if incomplete_files:
        logging.error("%s contains incomplete download artifacts:", name)
        for file in incomplete_files:
            logging.error("  %s", file)
        return False
    if not has_dataset_files(directory):
        logging.warning(
            "%s does not contain a standard dataset file extension. Manual inspection may be required.",
            name,
        )
    else:
        logging.info("%s contains recognizable dataset files.", name)
    logging.info("%s dataset integrity check passed.", name)
    return True

def verify_all_resources() -> bool:
    all_valid = True
    model_resources = [
        ("MiniCPM5-1B-SFT", MODELS_DIR / "MiniCPM5-1B-SFT"),
        ("SigLIP-HD", MODELS_DIR / "SigLIP-HD"),
    ]
    dataset_resources = [
        ("aya_dataset", DATASETS_DIR / "aya_dataset"),
        ("aya_collection_language_split", DATASETS_DIR / "aya_collection_language_split"),
        ("LLaVA-Pretrain", DATASETS_DIR / "LLaVA-Pretrain"),
    ]
    for name, directory in model_resources:
        if not verify_model_directory(name, directory):
            all_valid = False
    for name, directory in dataset_resources:
        if not verify_dataset_directory(name, directory):
            all_valid = False
    return all_valid

def calculate_sha256(file_path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()

def write_checksum_manifest(resources: list[tuple[str, Path]]) -> None:
    manifest_path = OUTPUT_DIR / "download_sha256.txt"
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for name, directory in resources:
            manifest.write(f"# {name}\n")
            for file in sorted(directory.rglob("*")):
                if not file.is_file():
                    continue
                relative_path = file.relative_to(directory)
                if ".hfd" in relative_path.parts:
                    continue
                sha256 = calculate_sha256(file)
                manifest.write(f"{sha256} {relative_path}\n")
            manifest.write("\n")
    logging.info("SHA-256 manifest written to: %s", manifest_path)

def main() -> int:
    setup_logging()
    logging.info("========== ETET Step 2: Download ==========")
    logging.info("Project root: %s", PROJECT_ROOT)
    logging.info("Python: %s", sys.version.replace("\n", " "))
    logging.info("Python executable: %s", sys.executable)
    logging.info("HF mirror: %s", HF_MIRROR)

    try:
        prepare_directories()
        hfd_path = find_hfd()
        logging.info("Using hfd.sh: %s", hfd_path)

        # Download MiniCPM5-1B-SFT from ModelScope.
        download_modelscope_model(MINICPM_REPO, MODELS_DIR / "MiniCPM5-1B-SFT")

        # Download SigLIP-HD from hf-mirror.
        download_huggingface(SIGLIP_REPO, MODELS_DIR / "SigLIP-HD", hfd_path, dataset=False)

        # Download Aya Dataset from ModelScope.
        download_modelscope_dataset(AYA_DATASET_REPO, DATASETS_DIR / "aya_dataset")

        # Download Aya Collection (language split) from hf-mirror.
        download_huggingface(
            AYA_COLLECTION_REPO,
            DATASETS_DIR / "aya_collection_language_split",
            hfd_path,
            dataset=True,
            include_patterns=[
                "*english*", "*English*",
                "*simplified_chinese*", "*Simplified_Chinese*",
                "*traditional_chinese*", "*Traditional_Chinese*",
            ],
            force=True,
        )

        # Download LLaVA-Pretrain from hf-mirror.
        download_huggingface(LLAVA_PRETRAIN_REPO, DATASETS_DIR / "LLaVA-Pretrain", hfd_path, dataset=True)

        logging.info("Starting resource integrity verification...")
        if not verify_all_resources():
            raise RuntimeError("One or more resource integrity checks failed.")

        resources = [
            ("MiniCPM5-1B-SFT", MODELS_DIR / "MiniCPM5-1B-SFT"),
            ("SigLIP-HD", MODELS_DIR / "SigLIP-HD"),
            ("aya_dataset", DATASETS_DIR / "aya_dataset"),
            ("aya_collection_language_split", DATASETS_DIR / "aya_collection_language_split"),
            ("LLaVA-Pretrain", DATASETS_DIR / "LLaVA-Pretrain"),
        ]
        write_checksum_manifest(resources)

        logging.info("========== ETET Step 2 completed successfully ==========")
        logging.info("All resources are ready.")
        logging.info("Models directory: %s", MODELS_DIR)
        logging.info("Datasets directory: %s", DATASETS_DIR)
        logging.info("Log file: %s", LOG_FILE)
        return 0

    except Exception as error:
        logging.exception("ETET Step 2 failed: %s", error)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
