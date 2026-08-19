#!/usr/bin/env bash
#
# Build the episodic memory DAG for one or more persons.
#
# Pipeline (per person)
# ---------------------
#   1. build_action        30-sec captions -> atomic action nodes
#                          -> output/memory/{source}/{person}/captions/{raw,action}.json
#   2. build_activity      captions -> activity nodes
#                          (segment -> retry -> disentangle -> merge)
#                          -> .../captions/{activity_raw,activity,activity_window_log}.json
#   3. build_conversation  captions -> conversation-topic nodes
#                          -> .../captions/conversation.json
#   4. build               assemble the DAG
#                          -> output/memory/{source}/{person}/dag.json
#
# Inputs are read from EGOCITE_DATA, prompts from EGOCITE_PROMPT, and everything
# written lands under EGOCITE_OUTPUT — see src/config.py.
#
# Usage
# -----
#   bash src/scripts/2-build_memory.sh [OPTIONS] [PERSON ...]
#
# Options
# -------
#   --source SOURCE       densecaption (default) | vlm
#                         densecaption = the released EgoLife dense captions
#                         vlm          = captions from the VLM captioner
#   --llm-name NAME       LLM for the activity and conversation stages.
#                         Default: gpt-5.4-2026-03-05
#   --action-llm-name N   LLM for action extraction. Default: same as --llm-name
#   --until-day N         Only process captions up to DAY N (action stage)
#   --skip-action | --skip-activity | --skip-conversation
#                         Skip a stage and reuse what is already on disk
#
# Model names route to a backend in src/models/__init__.py, and each backend
# reads its credentials from the "Credentials and endpoints" section of
# src/config.py:
#   chatgpt/* , *-codex    LiteLLM/Codex proxy   LITELLM_BASE_URL / LITELLM_API_KEY
#   gpt / o1 / o3 / o4     api.openai.com        OPENAI_API_KEY
#   anything else          local vLLM server     VLLM_BASE_URL
# Each stage batches its calls through the OpenAI Batch API when the backend
# supports it, and falls back to concurrent real-time calls when it does not.
#
# Environment
# -----------
#   EGOCITE_PYTHON        interpreter to run the stages with (default: python)
#
# Examples
# --------
#   bash src/scripts/2-build_memory.sh A1_JAKE
#   bash src/scripts/2-build_memory.sh --source vlm --llm-name chatgpt/gpt-5.4 A3_TASHA A5_KATRINA
#   bash src/scripts/2-build_memory.sh --skip-action --skip-activity A1_JAKE   # conversation + DAG only
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(dirname "$SCRIPT_DIR")"
MEM="$SRC_DIR/memory"

PY="${EGOCITE_PYTHON:-python}"
export PYTHONUNBUFFERED=1

SOURCE="densecaption"
LLM_NAME="gpt-5.4-2026-03-05"
ACTION_LLM_NAME=""
UNTIL_DAY=""
SKIP_ACTION=""
SKIP_ACTIVITY=""
SKIP_CONVERSATION=""
PERSONS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --source)            SOURCE="$2";          shift 2 ;;
        --llm-name)          LLM_NAME="$2";        shift 2 ;;
        --action-llm-name)   ACTION_LLM_NAME="$2"; shift 2 ;;
        --until-day)         UNTIL_DAY="$2";       shift 2 ;;
        --skip-action)       SKIP_ACTION=1;        shift ;;
        --skip-activity)     SKIP_ACTIVITY=1;      shift ;;
        --skip-conversation) SKIP_CONVERSATION=1;  shift ;;
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
[[ -z "$ACTION_LLM_NAME" ]] && ACTION_LLM_NAME="$LLM_NAME"
UNTIL_DAY_ARG=""
[[ -n "$UNTIL_DAY" ]] && UNTIL_DAY_ARG="--until-day $UNTIL_DAY"

FAILED=()

for PERSON in "${PERSONS[@]}"; do
    echo ""
    echo "========================================"
    echo "  Person        : $PERSON"
    echo "  Source        : $SOURCE"
    echo "  Action LLM    : $ACTION_LLM_NAME"
    echo "  Activity/Conv : $LLM_NAME"
    echo "  Start         : $(date +%H:%M:%S)"
    echo "========================================"

    if [[ -z "$SKIP_ACTION" ]]; then
        echo ""; echo "--- 1/4 Action extraction ---"
        $PY "$MEM/build_action.py" --person "$PERSON" --source "$SOURCE" \
            --action-llm-name "$ACTION_LLM_NAME" $UNTIL_DAY_ARG \
            || { echo "ACTION FAILED: $PERSON" >&2; FAILED+=("$PERSON:action"); continue; }
    fi

    if [[ -z "$SKIP_ACTIVITY" ]]; then
        echo ""; echo "--- 2/4 Activity (segment + retry + disentangle + merge) ---"
        $PY "$MEM/build_activity.py" --person "$PERSON" --source "$SOURCE" \
            --activity-llm-name "$LLM_NAME" \
            || { echo "ACTIVITY FAILED: $PERSON" >&2; FAILED+=("$PERSON:activity"); continue; }
    fi

    if [[ -z "$SKIP_CONVERSATION" ]]; then
        echo ""; echo "--- 3/4 Conversation grouping ---"
        $PY "$MEM/build_conversation.py" --person "$PERSON" --source "$SOURCE" \
            --conv-llm-name "$LLM_NAME" \
            || { echo "CONVERSATION FAILED: $PERSON" >&2; FAILED+=("$PERSON:conversation"); continue; }
    fi

    echo ""; echo "--- 4/4 Assemble DAG ---"
    $PY "$MEM/build.py" --person "$PERSON" --source "$SOURCE" \
        || { echo "DAG BUILD FAILED: $PERSON" >&2; FAILED+=("$PERSON:dag"); continue; }

    echo ""
    echo "  Done $(date +%H:%M:%S) -> output/memory/$SOURCE/$PERSON/dag.json"
done

echo ""
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "=== Finished $(date +%H:%M:%S) with ${#FAILED[@]} failure(s): ${FAILED[*]} ==="
    exit 1
fi
echo "=== All done $(date +%H:%M:%S) ==="
