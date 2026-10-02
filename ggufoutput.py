from __future__ import annotations
import argparse, json, logging, shutil, subprocess, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"
GGUF_DIR = MODELS_DIR / "gguf"
LLAMA_DIR = PROJECT_ROOT / "llama.cpp"
CONVERT_SCRIPT = LLAMA_DIR / "convert_hf_to_gguf.py"
QUANTIZE_BIN = LLAMA_DIR / "build" / "bin" / "llama-quantize"
SIMPLE_BIN = LLAMA_DIR / "build" / "bin" / "llama-simple"
GGUF_SRC_DIRNAME = "gguf_src"
ETET_ARCH = "ETETMOE_LLAMA"

# format -> (convert --outtype, llama-quantize type or None)
# all quants are produced by llama-quantize from the reused f16 base
QUANT_FORMATS = {
    "FP16":   ("f16", None),
    "Q8_0":   ("f16", "Q8_0"),
    "Q4_K_M": ("f16", "Q4_K_M"),
}

TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "special_tokens_map.json",
    "chat_template.jinja",
    "generation_config.json",
]

# smoke test loads each artifact with llama-simple and generates a few tokens
SMOKE_PROMPT = "The capital of France is"
SMOKE_TOKENS = 12
SMOKE_TIMEOUT = 300


def setup_logging():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("etet_ggufoutput")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)
    fh = logging.FileHandler(OUTPUT_DIR / "ggufoutput_output.log", mode="a", encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


logger = setup_logging()


def find_etet_models() -> list[Path]:
    # a directory counts as an ETET model when its config declares the ETET architecture
    found = []
    for d in sorted(MODELS_DIR.iterdir()):
        if not d.is_dir():
            continue
        for cfg in [d / "config.json", d / "language_model" / "config.json"]:
            if not cfg.is_file():
                continue
            try:
                data = json.loads(cfg.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if ETET_ARCH in (data.get("architectures") or []):
                found.append(d)
                break
    return found


def prepare_gguf_src(model_dir: Path) -> Path:
    # gguf_src mirrors the text model into one flat dir the converter expects
    # symlinks keep the large weight files out of the copy, the original dir is never touched
    src = model_dir / GGUF_SRC_DIRNAME
    if src.exists():
        shutil.rmtree(src)
    src.mkdir(parents=True)

    lm_dir = model_dir / "language_model"
    weight_dir = lm_dir if (lm_dir / "config.json").is_file() else model_dir
    roots = [model_dir, weight_dir]

    for p in weight_dir.glob("*.safetensors*"):
        (src / p.name).symlink_to(p.resolve())

    for name in TOKENIZER_FILES:
        for root in roots:
            p = root / name
            if p.is_file() and not (src / name).exists():
                (src / name).symlink_to(p.resolve())

    shutil.copy2(weight_dir / "config.json", src / "config.json")
    return src


def select_model() -> Path:
    models = find_etet_models()
    if not models:
        raise SystemExit(f"no ETET model found under {MODELS_DIR}")
    print("Available ETET models:")
    for i, d in enumerate(models, 1):
        print(f"  {i}. {d.name}")
    choice = input("Select model index: ").strip()
    try:
        return models[int(choice) - 1]
    except (ValueError, IndexError):
        raise SystemExit("invalid selection")


def select_format() -> list[str]:
    keys = list(QUANT_FORMATS.keys())
    print("Available quantization formats:")
    for i, k in enumerate(keys, 1):
        print(f"  {i}. {k}")
    print("  all. export every format")
    choice = input("Select quantization format: ").strip().lower()
    if choice == "all":
        return keys
    try:
        return [keys[int(choice) - 1]]
    except (ValueError, IndexError):
        raise SystemExit("invalid selection")


def run_convert(src: Path, outfile: Path, outtype: str, mmproj: bool, modelname: str) -> bool:
    cmd = [sys.executable, str(CONVERT_SCRIPT), str(src), "--outfile", str(outfile), "--outtype", outtype, "--model-name", modelname]
    if mmproj:
        cmd.append("--mmproj")
    logger.info(f"convert command: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"conversion failed (exit code {e.returncode})")
    except FileNotFoundError:
        logger.error(f"converter not found: {CONVERT_SCRIPT}")
    return False


def run_quantize(src_gguf: Path, out_gguf: Path, quant_type: str) -> bool:
    if not QUANTIZE_BIN.is_file():
        logger.warning(f"llama-quantize not found ({QUANTIZE_BIN}), build llama.cpp first, skipping {quant_type}")
        return False
    cmd = [str(QUANTIZE_BIN), str(src_gguf), str(out_gguf), quant_type]
    logger.info(f"quantize command: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"quantization failed (exit code {e.returncode})")
    return False


def run_smoke(gguf: Path) -> bool:
    if not SIMPLE_BIN.is_file():
        logger.warning(f"llama-simple not found ({SIMPLE_BIN}), build llama.cpp to enable the smoke test")
        return False
    cmd = [str(SIMPLE_BIN), "-m", str(gguf), "-n", str(SMOKE_TOKENS), SMOKE_PROMPT]
    logger.info(f"smoke command: {' '.join(cmd)}")
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=SMOKE_TIMEOUT)
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.error(f"smoke test failed: {e}")
        return False
    out = res.stdout.strip()
    ok = res.returncode == 0 and len(out) > 0
    logger.info(f"smoke test {'passed' if ok else 'failed'}: {gguf.name} -> {out[-80:]!r}")
    return ok


def report_output(path: Path):
    if path.is_file():
        size_mb = path.stat().st_size / (1024 * 1024)
        logger.info(f"artifact: {path} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="ETET GGUF packaging script")
    parser.add_argument("--model", type=str, default=None, help="model directory under models/, asked interactively when omitted")
    parser.add_argument("--format", type=str, default=None, choices=list(QUANT_FORMATS.keys()) + ["all"], help="quantization format, asked interactively when omitted")
    parser.add_argument("--mmproj", action="store_true", help="also export the multimodal projector file")
    parser.add_argument("--name", type=str, default=None, help="GGUF model name (general.name and file name), asked interactively when omitted")
    parser.add_argument("--no-smoke", action="store_true", help="skip the llama-simple smoke test on each artifact")
    args = parser.parse_args()

    logger.info("=== ETET GGUF packaging started ===")

    if args.model is not None:
        model_dir = MODELS_DIR / args.model
        if model_dir not in find_etet_models():
            raise SystemExit(f"{model_dir} is not a valid ETET model directory")
    else:
        model_dir = select_model()
    logger.info(f"selected model: {model_dir.name}")

    formats = [args.format] if args.format is not None and args.format != "all" else \
        list(QUANT_FORMATS.keys()) if args.format == "all" else select_format()

    # only ask for mmproj in interactive mode, non-interactive runs default to text-only
    mmproj = args.mmproj or (sys.stdin.isatty() and input("Also export mmproj (multimodal projector)? [y/N]: ").strip().lower() == "y")

    # the gguf model name is decided at pack time and never hardcoded; empty input falls back to the dir name
    if args.name is not None:
        modelname = args.name
    elif sys.stdin.isatty():
        modelname = input("Enter the GGUF model name (general.name and file name): ").strip()
    else:
        modelname = ""
    modelname = modelname or model_dir.name
    logger.info(f"model name: {modelname}")

    src = prepare_gguf_src(model_dir)
    logger.info(f"gguf_src ready: {src}")

    GGUF_DIR.mkdir(parents=True, exist_ok=True)

    # the f16 base is converted once and reused by every quant
    f16 = GGUF_DIR / f"{modelname}-f16.gguf"
    if f16.is_file():
        logger.info(f"reusing existing f16 base: {f16}")
    elif not run_convert(src, f16, QUANT_FORMATS["FP16"][0], mmproj=False, modelname=modelname):
        raise SystemExit("f16 base conversion failed, cannot continue with quantization")
    report_output(f16)
    if not args.no_smoke:
        run_smoke(f16)

    for fmt in formats:
        _, quant_type = QUANT_FORMATS[fmt]
        if quant_type is None:
            continue
        out = GGUF_DIR / f"{modelname}-{quant_type}.gguf"
        if run_quantize(f16, out, quant_type):
            report_output(out)
            if not args.no_smoke:
                run_smoke(out)

    if mmproj:
        mm_out = GGUF_DIR / f"{modelname}-f16.gguf"
        logger.info("trying mmproj export (available once the second stage mmproj converter lands)")
        if not run_convert(src, mm_out, QUANT_FORMATS["FP16"][0], mmproj=True, modelname=modelname):
            logger.warning("mmproj export failed, skipped; the text-only GGUF is unaffected")

    logger.info("=== ETET GGUF packaging finished ===")


if __name__ == "__main__":
    main()
