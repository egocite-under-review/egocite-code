#!/usr/bin/env bash
#
# Turn the raw EgoLife release into the 30-second caption list the memory build
# starts from, for one or more persons.
#
# Pipeline (per person)
# ---------------------
#   1. preprocess_egolife  DenseCaption SRTs -> translated jsonl -> per-clip sync
#                          -> output/preprocess/translated/{person}_*.jsonl
#                          -> output/preprocess/{source}_sync/{person}_*.json
#   2. caption_egolife     sync -> one first-person caption per video segment
#                          -> output/captions/{source}/{person}/{person}_captions.json
#
# That caption file is exactly what src/scripts/2-build_memory.sh reads next.
#
# Reads the EgoLife release from EGOCITE_EGOLIFE (default: $EGOCITE_DATASET/EgoLife)
# and writes everything under EGOCITE_OUTPUT — see src/config.py.
#
# Usage
# -----
#   bash src/scripts/1-preprocess_egolife.sh [OPTIONS] [PERSON ...]
#
# Options
# -------
#   --model NAME          LLM for translation and captioning.
#                         Default: Qwen/Qwen3.6-27B-FP8 (local vLLM server)
#   --source SOURCE       Caption set being produced: densecaption (default) | vlm
#   --data-dir DIR        EgoLife dataset root (default: config.EGOLIFE_DATASET)
#   --skip-translate      Reuse the existing translated/*.jsonl
#   --skip-sync           Skip sync generation
#   --skip-caption        Stop after sync (no captioning pass)
#   --overwrite           Regenerate a caption file that already exists
#
# Model names route to a backend in src/models/__init__.py and read their key
# from the "Credentials and endpoints" section of src/config.py.
#
# Environment
# -----------
#   EGOCITE_PYTHON        interpreter to run the stages with (default: python)
#
# Examples
# --------
#   bash src/scripts/1-preprocess_egolife.sh A1_JAKE
#   bash src/scripts/1-preprocess_egolife.sh --model gpt-5.4-mini A1_JAKE A2_ALICE
#   bash src/scripts/1-preprocess_egolife.sh --skip-translate --skip-sync A1_JAKE   # caption only
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(dirname "$SCRIPT_DIR")"
PRE="$SRC_DIR/preprocess"

PY="${EGOCITE_PYTHON:-python}"
export PYTHONUNBUFFERED=1

# Empty = let each stage pick its own default; caption_egolife.py chooses
# the model that matches --source (Gemma for gemma, Qwen otherwise).
MODEL=""
SOURCE="densecaption"
DATA_DIR=""
SKIP_TRANSLATE=""
SKIP_SYNC=""
SKIP_CAPTION=""
OVERWRITE=""
PERSONS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)          MODEL="$2";       shift 2 ;;
        --source)         SOURCE="$2";      shift 2 ;;
        --data-dir)       DATA_DIR="$2";    shift 2 ;;
        --skip-translate) SKIP_TRANSLATE=1; shift ;;
        --skip-sync)      SKIP_SYNC=1;      shift ;;
        --skip-caption)   SKIP_CAPTION=1;   shift ;;
        --overwrite)      OVERWRITE=1;      shift ;;
        -h|--help) sed -n '/^# Usage/,/^set -/p' "$0" | grep '^#' | sed 's/^# \?//'; exit 0 ;;
        -*) echo "Unknown option: $1" >&2; exit 1 ;;
        *)  PERSONS+=("$1"); shift ;;
    esac
done

case "$SOURCE" in
    densecaption|gemini|gemma) ;;
    *) echo "--source must be 'densecaption', 'gemini' or 'gemma' (got '$SOURCE')" >&2; exit 1 ;;
esac

[[ ${#PERSONS[@]} -eq 0 ]] && PERSONS=(A1_JAKE A2_ALICE A3_TASHA A4_LUCIA A5_KATRINA A6_SHURE)

PRE_ARGS=()
[[ -n "$MODEL"          ]] && PRE_ARGS+=(--model "$MODEL")
[[ -n "$DATA_DIR"       ]] && PRE_ARGS+=(--data-dir "$DATA_DIR")
[[ -n "$SKIP_TRANSLATE" ]] && PRE_ARGS+=(--skip-translate)
[[ -n "$SKIP_SYNC"      ]] && PRE_ARGS+=(--skip-sync)
CAP_ARGS=()
[[ -n "$MODEL"     ]] && CAP_ARGS+=(--model "$MODEL")
[[ -n "$OVERWRITE" ]] && CAP_ARGS+=(--overwrite)

FAILED=()

for PERSON in "${PERSONS[@]}"; do
    echo ""
    echo "========================================"
    echo "  Person : $PERSON"
    echo "  Source : $SOURCE"
    echo "  Model  : ${MODEL:-auto (matches --source)}"
    echo "  Start  : $(date +%H:%M:%S)"
    echo "========================================"

    if [[ -z "$SKIP_TRANSLATE" || -z "$SKIP_SYNC" ]]; then
        echo ""; echo "--- 1/2 Translate + sync ---"
        $PY "$PRE/preprocess_egolife.py" --person "$PERSON" \
            --source "$SOURCE" "${PRE_ARGS[@]}" \
            || { echo "PREPROCESS FAILED: $PERSON" >&2; FAILED+=("$PERSON:preprocess"); continue; }
    fi

    if [[ -z "$SKIP_CAPTION" ]]; then
        echo ""; echo "--- 2/2 Captioning ---"
        $PY "$PRE/caption_egolife.py" --person "$PERSON" \
            --source "$SOURCE" "${CAP_ARGS[@]}" \
            || { echo "CAPTION FAILED: $PERSON" >&2; FAILED+=("$PERSON:caption"); continue; }
    fi

    echo ""
    echo "  Done $(date +%H:%M:%S) -> output/captions/$SOURCE/$PERSON/${PERSON}_captions.json"
done

echo ""
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "=== Finished $(date +%H:%M:%S) with ${#FAILED[@]} failure(s): ${FAILED[*]} ==="
    exit 1
fi
echo "=== All done $(date +%H:%M:%S) ==="
