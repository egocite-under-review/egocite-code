#!/usr/bin/env bash
#
# install.sh — one command to get a working EgoCITE environment.
#
#   bash install.sh                 # conda env named "egocite"
#   bash install.sh myenv           # a different name
#   EGOCITE_NO_GPU=1 bash install.sh    # skip the GPU extras
#
# What it does
#   1. conda env with python 3.12 + ffmpeg (needed by the video captioner)
#   2. torch — the CUDA build when an NVIDIA GPU is present, else CPU
#   3. requirements.txt (the 15 packages EgoCITE imports)
#   4. flash-attn from a prebuilt wheel (cu128 / torch 2.8 / cp312). REQUIRED:
#      EmbeddingModel asks for flash_attention_2 with no fallback, so retrieval
#      cannot run without it — and it needs an NVIDIA GPU
#   5. an import check so you know it worked before you run anything
#
# Everything is pinned in requirements.txt. Re-running is safe.
set -euo pipefail

ENV_NAME="${1:-egocite}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

step() { echo ""; echo "=== $* ==="; }

command -v conda >/dev/null 2>&1 || {
    echo "conda not found. Install Miniconda first: https://docs.conda.io/en/latest/miniconda.html" >&2
    exit 1
}
eval "$(conda shell.bash hook)"

# ---- 1. environment -------------------------------------------------------
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    step "conda env '$ENV_NAME' already exists — reusing it"
else
    step "Creating conda env '$ENV_NAME' (python 3.12 + ffmpeg)"
    conda create -y -n "$ENV_NAME" -c conda-forge python=3.12 ffmpeg
fi
conda activate "$ENV_NAME"
python -V

# ---- 2. torch -------------------------------------------------------------
# PINNED: the prebuilt flash-attn wheel below is compiled against this exact
# torch, and a mismatch fails at import with "undefined symbol". Bump both
# together or not at all.
TORCH_VERSION="2.8.0"
HAS_GPU=""
if [[ -z "${EGOCITE_NO_GPU:-}" ]] && command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    HAS_GPU=1
fi
step "Installing torch $TORCH_VERSION ($([[ -n "$HAS_GPU" ]] && echo "CUDA 12.8 build — GPU detected" || echo "CPU build"))"
if [[ -n "$HAS_GPU" ]]; then
    pip install "torch==${TORCH_VERSION}" --index-url https://download.pytorch.org/whl/cu128
else
    pip install "torch==${TORCH_VERSION}" --index-url https://download.pytorch.org/whl/cpu
fi

# ---- 3. the pinned dependencies ------------------------------------------
step "Installing requirements.txt"
pip install -r requirements.txt

# ---- 4. optional GPU extras ----------------------------------------------
# Prebuilt flash-attn wheel matching this stack (cu128 / torch 2.8 / cp312 /
# linux x86_64) — compiling from source takes tens of minutes. If the wheel does
# not match the machine, fall back to a source build.
FLASH_ATTN_WHEEL="https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.8.3+cu128torch2.8-cp312-cp312-linux_x86_64.whl"

if [[ -n "$HAS_GPU" ]]; then
    step "Installing the CUDA toolkit into the env (nvcc — needed to build flash-attn)"
    if [[ ! -x "$CONDA_PREFIX/bin/nvcc" ]]; then
        conda install -y -c nvidia cuda-toolkit \
            || echo "  cuda-toolkit install failed — a source build of flash-attn will not work."
    else
        echo "  nvcc already present: $($CONDA_PREFIX/bin/nvcc --version | tail -1)"
    fi

    # flash-attn's setup.py reads CUDA_HOME. With a conda-installed toolkit that
    # is the env prefix; a system toolkit usually lives in /usr/local/cuda.
    if [[ -x "$CONDA_PREFIX/bin/nvcc" ]]; then
        export CUDA_HOME="$CONDA_PREFIX"
    elif [[ -x /usr/local/cuda/bin/nvcc ]]; then
        export CUDA_HOME=/usr/local/cuda
    fi
    export PATH="${CUDA_HOME:-$CONDA_PREFIX}/bin:$PATH"
    echo "  CUDA_HOME=${CUDA_HOME:-<unset>}"

    step "Installing flash-attn (REQUIRED — the embedding model has no fallback)"
    pip install "$FLASH_ATTN_WHEEL" \
        || pip install flash-attn --no-build-isolation \
        || echo "  flash-attn FAILED to install — retrieval will not work."
    pip install faiss-gpu-cu12 || \
        echo "  faiss-gpu-cu12 unavailable — keeping faiss-cpu."
fi

# ---- 5. test --------------------------------------------------------------
# Offline and free: no API key, no vLLM server, no dataset. Catches the things
# that fail silently — a missing dependency, a prompt file that moved, a config
# path that will not resolve — before you spend a run finding out.
step "Testing the installation"
python - <<'PY'
import importlib, json, os, subprocess, sys, tempfile
sys.path.insert(0, "src")
P, F = [], []

def check(name, fn):
    try:
        detail = fn() or ""
        P.append(name); print(f"  ok    {name}{'  — ' + detail if detail else ''}")
    except Exception as e:
        F.append(name); print(f"  FAIL  {name}\n          {type(e).__name__}: {e}")

def deps():
    missing = []
    for m in ("igraph", "pydantic", "numpy", "tqdm", "pqdm", "faiss",
              "sentence_transformers", "accelerate", "transformers", "openai",
              "anthropic", "google.genai", "tenacity", "PIL", "pandas", "pysrt",
              "huggingface_hub", "torch"):
        try:
            importlib.import_module(m)
        except Exception as e:
            missing.append(f"{m} ({type(e).__name__})")
    if missing:
        raise ImportError("; ".join(missing))
    return "18 packages"

def attention():
    # EmbeddingModel hardcodes attn_implementation="flash_attention_2" with no
    # fallback, so a missing or ABI-mismatched flash-attn means no retrieval.
    import torch
    note = f"torch {torch.__version__}, cuda={torch.cuda.is_available()}"
    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device — flash_attention_2 requires an NVIDIA GPU")
    try:
        import flash_attn
    except ImportError as e:
        import importlib.metadata as md
        try:
            md.version("flash_attn")
            raise RuntimeError(f"flash-attn is installed but UNUSABLE against torch "
                               f"{torch.__version__} ({e}). Reinstall one to match the other.")
        except md.PackageNotFoundError:
            raise RuntimeError("flash-attn is not installed, and the embedding model "
                               "has no fallback — retrieval will fail on every search.")
    return note + f", flash-attn {flash_attn.__version__}"

def pkg():
    import config, prompts                              # noqa: F401
    from agent import EpisodicAgent, EpisodicSearch     # noqa: F401
    from memory import EpisodicDAG                      # noqa: F401
    from models import make_llm                         # noqa: F401
    return "config, prompts, agent, memory, models"

def paths():
    import config
    for n in ("DATA_DIR", "DATASET_DIR", "OUTPUT_DIR", "PROMPT_DIR", "EGOLIFE_DATASET",
              "EGOLIFEQA_DATASET", "EGOR1_DATASET", "EGOMEM_DATASET"):
        if not getattr(config, n, None):
            raise ValueError(f"{n} is empty")
    if not os.path.isdir(config.PROMPT_DIR):
        raise FileNotFoundError(f"PROMPT_DIR missing: {config.PROMPT_DIR}")
    return "8 roots"

def prompt_files():
    import glob, config, prompts
    names = [os.path.relpath(p, config.PROMPT_DIR)[:-4].replace(os.sep, "/")
             for p in glob.glob(os.path.join(config.PROMPT_DIR, "**", "*.txt"), recursive=True)]
    if len(names) < 20:
        raise ValueError(f"only {len(names)} prompt files found")
    allowed_empty = {"agent/retrieval_nocurate/curate_tool_entry"}
    for n in names:
        if not prompts.load(n, strip=False) and n not in allowed_empty:
            raise ValueError(f"empty prompt: {n}")
    return f"{len(names)} files"

def retrieval_prompt():
    import agent.agent as A
    sizes = []
    for use_curate in (True, False):
        s = A._build_tool_use_system(use_curate)
        if "$" in s:
            raise ValueError(f"unsubstituted placeholder (use_curate={use_curate})")
        if "search_action_speech" not in s:
            raise ValueError("prompt body missing")
        if use_curate != ("curate_evidence" in s):
            raise ValueError(f"wrong fragment set for use_curate={use_curate}")
        sizes.append(len(s))
    return f"{sizes[0]} / {sizes[1]} chars"

def curation_cap():
    import agent.agent as A, prompts
    cap = A._CURATE_EVIDENCE_THRESHOLD
    text = (prompts.load("agent/curation_system")
            + prompts.load("agent/retrieval_curate/curate_tool_entry", strip=False))
    if f"5 and {cap}" not in text:
        raise ValueError(f"prompts do not state the cap of {cap}")
    return f"{cap} items"

def dag_roundtrip():
    from memory import EpisodicDAG
    d = EpisodicDAG()
    a = d.add_node(level="activity", text="I cook dinner", start_time="18000000",
                   end_time="18100000", date="DAY1", raw_caption_indices=[0, 1])
    b = d.add_node(level="action", text="I chop an onion", start_time="18000000",
                   end_time="18003000", date="DAY1", raw_caption_indices=[0])
    d.add_edge(a, b)
    s = d.summary()
    if s["nodes"] != 2 or s["edges"] != 1:
        raise ValueError(f"unexpected summary: {s}")
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "dag.json"); d.save(p); data = json.load(open(p))
    if len(data["vertices"]) != 2 or data["edges"] != [[a, b]]:
        raise ValueError("saved DAG does not match what was built")
    return "2 nodes, 1 edge"

def helpers():
    from memory.segmentation import _time_to_sec, _date_to_day, abs_sec, SegmentationOutput
    assert _time_to_sec("11094300") == 11*3600 + 9*60 + 43, "_time_to_sec"
    assert _date_to_day("DAY3") == 2, "_date_to_day"
    assert abs_sec("DAY2", "00010000") == 86400 + 60, "abs_sec"   # HHMMSSFF
    g = SegmentationOutput.model_validate_json('{"groups": [{"indices": ["0", 1], "summary": "x"}]}')
    assert g.groups[0].indices == [0, 1], "schema coercion"
    return "time helpers + schema"

def harnesses():
    for h in ("eval_egolifeqa.py", "eval_egomem.py", "eval_egor1.py"):
        r = subprocess.run([sys.executable, os.path.join("src", "eval", h), "--help"],
                           capture_output=True, text=True, timeout=180)
        if r.returncode != 0:
            raise RuntimeError(f"{h}: {r.stderr.strip()[:200]}")
    return "3 harnesses"

def shell_scripts():
    import glob
    for s in sorted(glob.glob("src/scripts/*.sh")) + ["install.sh"]:
        r = subprocess.run(["bash", "-n", s], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"{os.path.basename(s)}: {r.stderr.strip()[:200]}")
    return "5 scripts"

for name, fn in (("third-party imports", deps),
                 ("torch / attention backend", attention),
                 ("EgoCITE package imports", pkg),
                 ("config paths resolve", paths),
                 ("every prompt file loads", prompt_files),
                 ("retrieval prompt renders (both tool sets)", retrieval_prompt),
                 ("curation cap consistent in code and prompts", curation_cap),
                 ("DAG round-trips through JSON", dag_roundtrip),
                 ("segmentation helpers", helpers),
                 ("eval harnesses respond to --help", harnesses),
                 ("shell scripts parse", shell_scripts)):
    check(name, fn)

# Data and built memories are optional at install time — reported, never failed.
import config
print("\n  data (optional, not part of the result):")
for label, path in (("EgoLife release", config.EGOLIFE_DATASET),
                    ("EgoLifeQA      ", os.path.join(config.EGOLIFEQA_DATASET, "Ego-QA-4.4K")),
                    ("Ego-R1-Bench   ", config.EGOR1_DATASET),
                    ("EgoMem         ", os.path.join(config.EGOMEM_DATASET, "data"))):
    print(f"    {'present' if os.path.exists(path) else 'missing'}  {label}  {path}")
built = [f"{s}/{p}" for s in config.SOURCES for p in config.PERSONS
         if os.path.exists(config.dag_file(p, s))]
print(f"    {len(built)} built memories under {config.MEMORY_DIR}")

print(f"\n  {len(P)} passed, {len(F)} failed")
sys.exit(1 if F else 0)
PY

ffmpeg -version 2>/dev/null | head -1 || echo "  WARNING: ffmpeg missing (only caption_video.py needs it)"

cat <<EOF

=== Done. ===

  conda activate $ENV_NAME

Next:
  bash src/scripts/0-download.sh             # fetch the datasets
  export EGOCITE_DATASET=$HERE/dataset
  bash src/scripts/1-preprocess_egolife.sh A1_JAKE
  bash src/scripts/2-build_memory.sh A1_JAKE
  bash src/scripts/3-eval.sh --bench egolifeqa A1_JAKE

API keys go in src/config.py (or the matching env vars).
EOF
