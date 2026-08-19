"""
Single source of truth for every path EgoCITE touches.

Separation of concerns:
  INPUTS  (raw captions, benchmark question files) live under ``DATA_DIR`` /
          ``DATASET_DIR`` and are never written to.
  PROMPTS live under ``PROMPT_DIR`` as .txt files (see ``prompts.py``).
  OUTPUTS (built memories, run logs, eval results, figures) all land under
          ``OUTPUT_DIR`` — nothing is written next to the inputs.

Every location is set here and can be overridden with an environment variable,
so the same code runs against a different data drop without edits:

    EGOCITE_EGOLIFE  raw EgoLife release       (default: $EGOCITE_DATASET/EgoLife)
    EGOCITE_EGOLIFEQA  EgoLifeQA question set  (default: $EGOCITE_DATASET/Ego-R1-Data)
    EGOCITE_EGOR1      Ego-R1-Bench            (default: $EGOCITE_DATASET/Ego-R1-Bench)
    EGOCITE_EGOMEM     EgoMem question set     (default: $EGOCITE_DATASET/LifeDialBench)
    EGOCITE_DATA     pre-built caption inputs  (default: ../long-memory/data)
    EGOCITE_DATASET  benchmark question files  (default: ../dataset)
    EGOCITE_OUTPUT   all generated content     (default: EgoCITE/output)
    EGOCITE_PROMPT   prompt .txt files         (default: EgoCITE/prompt)
"""

import os

# EgoCITE/src/config.py -> EgoCITE/
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)

# Where the original project kept its inputs. Only raw captions are read from here.
_DEFAULT_DATA = os.path.join(os.path.dirname(PROJECT_ROOT), "long-memory", "data")
# Datasets live inside the project: EgoCITE/dataset/, which 0-download.sh fills.
_DEFAULT_DATASET = os.path.join(PROJECT_ROOT, "dataset")

DATA_DIR = os.environ.get("EGOCITE_DATA", _DEFAULT_DATA)
DATASET_DIR = os.environ.get("EGOCITE_DATASET", _DEFAULT_DATASET)
OUTPUT_DIR = os.environ.get("EGOCITE_OUTPUT", os.path.join(PROJECT_ROOT, "output"))
PROMPT_DIR = os.environ.get("EGOCITE_PROMPT", os.path.join(PROJECT_ROOT, "prompt"))

# The raw EgoLife release, read by src/preprocess/ and never written to. Point
# this at wherever the download lives — the layout it expects is:
#     EGOLIFE_DATASET/EgoLifeCap/DenseCaption/{person}/DAYn/*.srt
#     EGOLIFE_DATASET/EgoLifeCap/Transcript/{person}/DAYn/*.srt
#     EGOLIFE_DATASET/{person}/DAYn/*.mp4          (the video clips)
EGOLIFE_DATASET = os.environ.get("EGOCITE_EGOLIFE",
                                 os.path.join(DATASET_DIR, "EgoLife"))

# Benchmark question sets, one root per benchmark — the directory names on disk
# do not match the benchmark names, so they are spelled out here rather than
# buried in the eval scripts. Read by src/eval/ only.
#   EgoLifeQA     Ego-R1-Data/Ego-QA-4.4K/manual-2.9K/{person}.json
#   Ego-R1-Bench  Ego-R1-Bench/{manual|gemini}-benchmark/{person}.json
#   EgoMem        LifeDialBench/data/EgoMem-Normalized.json
EGOLIFEQA_DATASET = os.environ.get("EGOCITE_EGOLIFEQA",
                                   os.path.join(DATASET_DIR, "Ego-R1-Data"))
EGOR1_DATASET = os.environ.get("EGOCITE_EGOR1",
                               os.path.join(DATASET_DIR, "Ego-R1-Bench"))
EGOMEM_DATASET = os.environ.get("EGOCITE_EGOMEM",
                                os.path.join(DATASET_DIR, "LifeDialBench"))

MEMORY_DIR = os.path.join(OUTPUT_DIR, "memory")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
RESULT_DIR = os.path.join(OUTPUT_DIR, "results")
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")

PERSONS = ["A1_JAKE", "A2_ALICE", "A3_TASHA", "A4_LUCIA", "A5_KATRINA", "A6_SHURE"]
# Every caption source the pipeline understands. A source names both where the
# 30-second captions come from and where all of its derived output lives, so
# two sources never overwrite each other.
#   densecaption  EgoLife's released DenseCaption SRTs (human-annotated)
#   gemini        VLM captioner, Gemini backend   (src/preprocess/caption_video.py --model gemini)
#   gemma         VLM captioner, local Gemma on vLLM (--model gemma)
SOURCES = ["densecaption", "gemini", "gemma"]
# Sources produced by the VLM captioner rather than by translating SRTs.
VLM_SOURCES = ["gemini", "gemma"]


# ===========================================================================
# Credentials and endpoints
# ===========================================================================
# This is the ONLY place in the project where an API key or a service URL is
# read. Nothing under src/models/ hardcodes one.
#
# Put your own key in the quotes below, or leave it empty and export the
# environment variable of the same name — the env var wins when both are set.
# A backend only needs the entries it actually uses; an empty value raises a
# clear error naming the setting the moment that backend is constructed.

# --- OpenAI ---------------------------------------------------------------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
# Leave empty for api.openai.com; set it to point at a compatible gateway.
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "")

# --- Anthropic ------------------------------------------------------------
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# --- Local vLLM server (Qwen) ---------------------------------------------
# OpenAI-compatible; serves the model named by --llm-name qwen. Needs no real
# key, but the OpenAI client insists on a non-empty string.
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8001/v1")
VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "EMPTY")

# --- LiteLLM proxy --------------------------------------------------------
# Used by the `chatgpt/*` and `*-codex` model names.
LITELLM_BASE_URL = os.environ.get("LITELLM_BASE_URL", "")
LITELLM_API_KEY = os.environ.get("LITELLM_API_KEY", "")
LITELLM_REASONING = os.environ.get("LITELLM_REASONING", "none")

# --- Video captioning (src/preprocess/caption_video.py) -------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
# A second, vision-capable vLLM server for the local captioning backend
# (`--model gemma`). Separate from VLLM_BASE_URL, which serves the text model
# the agent uses, so both can run at once on different ports.
VLLM_VISION_BASE_URL = os.environ.get("VLLM_VISION_BASE_URL",
                                      "http://localhost:8002/v1")
VLLM_VISION_MODEL = os.environ.get("VLLM_VISION_MODEL", "google/gemma-4-31B-it")

# --- HuggingFace (src/scripts/0-download.py) ------------------------------
# Only needed for gated dataset repos; `huggingface-cli login` works too.
HUGGINGFACE_TOKEN = os.environ.get("HF_TOKEN", "")


def require(name: str) -> str:
    """Return the credential/endpoint `name`, or raise if it is still empty."""
    value = globals().get(name, "")
    if not value:
        raise RuntimeError(
            f"{name} is not set. Put your value in the 'Credentials and "
            f"endpoints' section of {os.path.abspath(__file__)}, or export "
            f"{name} in the environment."
        )
    return value


def _ensure(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Inputs (read-only)
# ---------------------------------------------------------------------------

def raw_caption_file(person: str, source: str = "densecaption") -> str:
    """The per-person 30-second caption list that the memory build starts from.

    densecaption   : captions derived from EgoLife's released DenseCaption SRTs.
    gemini / gemma : captions generated by the VLM captioner on that backend.

    Prefers a file this project produced (see `caption_file`); falls back to a
    pre-built one under DATA_DIR when src/preprocess/ has not been run.
    """
    produced = caption_file(person, source)
    if os.path.exists(produced):
        return produced
    if source == "gemini":
        # The pre-built VLM captions shipped with the original project were
        # produced by the Gemini backend.
        return os.path.join(DATA_DIR, "vlm_captions", "captions", person,
                            f"{person}_captions.json")
    if source in VLM_SOURCES:
        return produced
    return os.path.join(DATA_DIR, "captions", person, f"{person}_captions.json")


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

def preprocess_dir(name: str, create: bool = False) -> str:
    """output/preprocess/{name}/ — an EgoLife preprocessing intermediate."""
    d = os.path.join(OUTPUT_DIR, "preprocess", name)
    return _ensure(d) if create else d


def translated_dir(create: bool = False) -> str:
    """output/preprocess/translated/ — the DenseCaption SRTs translated to
    English, one .jsonl per SRT. Densecaption-only by nature: the VLM captioner
    writes English directly, so there is nothing to translate."""
    return preprocess_dir("translated", create)


def sync_dir(source: str = "densecaption", create: bool = False) -> str:
    """output/preprocess/{source}_sync/ — captions + transcripts aligned to the
    video clips they belong to, one .json per clip-hour.

    densecaption -> densecaption_sync/   built from translated/
    gemini       -> gemini_sync/         built from the Gemini captioner's output
    gemma        -> gemma_sync/          built from the Gemma captioner's output
    """
    return preprocess_dir(f"{source}_sync", create)


def vlm_caption_dir(source: str, create: bool = False) -> str:
    """output/preprocess/vlm_video_caption/{source}/ — the raw per-day output of
    the VLM captioner, one directory per backend so gemini and gemma never mix."""
    return preprocess_dir(os.path.join("vlm_video_caption", source), create)


def caption_file(person: str, source: str = "densecaption", create: bool = False) -> str:
    """output/captions/{source}/{person}/{person}_captions.json — the 30-second
    caption list produced by src/preprocess/, and the input to the memory build."""
    d = os.path.join(OUTPUT_DIR, "captions", source, person)
    if create:
        _ensure(d)
    return os.path.join(d, f"{person}_captions.json")


def memory_dir(person: str, source: str = "densecaption", create: bool = False) -> str:
    """output/memory/{source}/{person}/ — one built memory."""
    d = os.path.join(MEMORY_DIR, source, person)
    return _ensure(d) if create else d


def captions_dir(person: str, source: str = "densecaption", create: bool = False) -> str:
    """Intermediate build artifacts: raw/action/activity/conversation .json."""
    d = os.path.join(memory_dir(person, source), "captions")
    return _ensure(d) if create else d


def dag_file(person: str, source: str = "densecaption") -> str:
    """The assembled episodic DAG — the memory the agent searches."""
    return os.path.join(memory_dir(person, source), "dag.json")


def log_file(name: str, subdir: str = "") -> str:
    """output/logs/[subdir/]{name} — run logs."""
    return os.path.join(_ensure(os.path.join(LOG_DIR, subdir)), name)


def result_file(name: str, subdir: str = "") -> str:
    """output/results/[subdir/]{name} — eval result JSON."""
    return os.path.join(_ensure(os.path.join(RESULT_DIR, subdir)), name)
