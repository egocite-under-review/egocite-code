#!/usr/bin/env bash
#
# Run a benchmark over one or more persons.
#
# Picks the harness in src/eval/ that matches --bench, runs it once per person,
# and reports which runs failed:
#
#   egolifeqa   EgoLifeQA      src/eval/eval_egolifeqa.py
#   egomem      EgoMem         src/eval/eval_egomem.py
#   egor1       Ego-R1-Bench   src/eval/eval_egor1.py  — BOTH splits per person
#                              (manual and gemini), so you only pass the person
#
# Results land in output/results/{bench}/{source}/ and logs in
# output/logs/{bench}/{source}/ — see src/config.py.
#
# Usage
# -----
#   bash src/scripts/3-eval.sh --bench BENCH [OPTIONS] [PERSON ...]
#
# Options
# -------
#   --bench BENCH ...     egolifeqa | egomem | egor1  (repeatable, required)
#   --source SOURCE       densecaption (default) | vlm — which memory to use
#   --llm-name NAME       sonnet | opus | qwen (default) | gpt
#                         selects the MODEL_CONFIG block in src/agent/agent.py
#   --all                 Run every person instead of the ones listed
#   --dry-run             Print the commands without running them
#   -- ARGS...            Everything after a bare `--` is passed to the harness
#
# Ego-R1-Bench ships two splits and both are run for every person. Pass
# `-- --benchmark manual` (or gemini) only if you want to restrict it to one.
#
# Environment
# -----------
#   EGOCITE_PYTHON        interpreter to run the harnesses with (default: python)
#
# Examples
# --------
#   bash src/scripts/3-eval.sh --bench egolifeqa A1_JAKE -- --n-end 20
#   bash src/scripts/3-eval.sh --bench egomem --all --source vlm
#   bash src/scripts/3-eval.sh --bench egolifeqa egomem egor1 --all --llm-name gpt
#   bash src/scripts/3-eval.sh --bench egor1 --all                 # both splits
#   bash src/scripts/3-eval.sh --bench egor1 A1_JAKE -- --benchmark gemini
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(dirname "$SCRIPT_DIR")"
EVAL="$SRC_DIR/eval"

PY="${EGOCITE_PYTHON:-python}"
export PYTHONUNBUFFERED=1

ALL_PERSONS=(A1_JAKE A2_ALICE A3_TASHA A4_LUCIA A5_KATRINA A6_SHURE)

BENCHES=()
PERSONS=()
EXTRA=()
SOURCE="densecaption"
LLM_NAME="qwen"
RUN_ALL=""
DRY_RUN=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --bench)
            # Consume only known benchmark names, so `--bench egomem A1_JAKE`
            # reads the person as a person and not as a second benchmark.
            shift
            while [[ $# -gt 0 ]]; do
                case "$1" in
                    egolifeqa|egomem|egor1) BENCHES+=("$1"); shift ;;
                    *) break ;;
                esac
            done ;;
        --source)    SOURCE="$2";   shift 2 ;;
        --llm-name)  LLM_NAME="$2"; shift 2 ;;
        --all)       RUN_ALL=1;     shift ;;
        --dry-run)   DRY_RUN=1;     shift ;;
        --)          shift; EXTRA+=("$@"); break ;;
        -h|--help) sed -n '/^# Usage/,/^set -/p' "$0" | grep '^#' | sed 's/^# \?//'; exit 0 ;;
        -*) echo "Unknown option: $1  (pass harness flags after a bare --)" >&2; exit 1 ;;
        *)  PERSONS+=("$1"); shift ;;
    esac
done

if [[ ${#BENCHES[@]} -eq 0 ]]; then
    echo "--bench is required: egolifeqa | egomem | egor1" >&2; exit 1
fi
for B in "${BENCHES[@]}"; do
    case "$B" in
        egolifeqa|egomem|egor1) ;;
        *) echo "Unknown benchmark '$B' (egolifeqa | egomem | egor1)" >&2; exit 1 ;;
    esac
done
case "$SOURCE" in
    densecaption|gemini|gemma) ;;
    *) echo "--source must be 'densecaption', 'gemini' or 'gemma' (got '$SOURCE')" >&2; exit 1 ;;
esac

[[ -n "$RUN_ALL" ]] && PERSONS=("${ALL_PERSONS[@]}")
[[ ${#PERSONS[@]} -eq 0 ]] && PERSONS=("${ALL_PERSONS[0]}")

# Ego-R1-Bench has two splits; run both unless the caller pinned one via `--`.
EGOR1_SPLITS=(manual gemini)
if [[ " ${EXTRA[*]-} " == *" --benchmark "* ]]; then
    EGOR1_SPLITS=("")
fi

N_RUNS=0
for BENCH in "${BENCHES[@]}"; do
    if [[ "$BENCH" == "egor1" ]]; then
        N_RUNS=$(( N_RUNS + ${#PERSONS[@]} * ${#EGOR1_SPLITS[@]} ))
    else
        N_RUNS=$(( N_RUNS + ${#PERSONS[@]} ))
    fi
done
echo "$N_RUNS run(s): bench=${BENCHES[*]}  persons=${PERSONS[*]}  source=$SOURCE  llm=$LLM_NAME"

FAILED=()

for BENCH in "${BENCHES[@]}"; do
    SPLITS=("")
    [[ "$BENCH" == "egor1" ]] && SPLITS=("${EGOR1_SPLITS[@]}")
    for PERSON in "${PERSONS[@]}"; do
        for SPLIT in "${SPLITS[@]}"; do
            LABEL="$BENCH"; SPLIT_ARG=()
            if [[ -n "$SPLIT" ]]; then
                LABEL="$BENCH/$SPLIT"; SPLIT_ARG=(--benchmark "$SPLIT")
            fi
            echo ""
            echo "========================================"
            echo "  Benchmark : $LABEL"
            echo "  Person    : $PERSON"
            echo "  Source    : $SOURCE"
            echo "  LLM       : $LLM_NAME"
            echo "  Start     : $(date +%H:%M:%S)"
            echo "========================================"
            CMD=("$PY" "$EVAL/eval_${BENCH}.py" --person "$PERSON" --source "$SOURCE"
                 --llm-name "$LLM_NAME" ${SPLIT_ARG[@]+"${SPLIT_ARG[@]}"}
                 ${EXTRA[@]+"${EXTRA[@]}"})
            echo "  ${CMD[*]}"
            [[ -n "$DRY_RUN" ]] && continue
            "${CMD[@]}" || { echo "FAILED: $LABEL $PERSON" >&2; FAILED+=("$LABEL:$PERSON"); }
        done
    done
done

echo ""
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "=== Finished $(date +%H:%M:%S) with ${#FAILED[@]} failure(s): ${FAILED[*]} ==="
    exit 1
fi
if [[ -z "$DRY_RUN" ]]; then
    for BENCH in "${BENCHES[@]}"; do
        echo "=== $BENCH done -> output/results/$BENCH/$SOURCE/ ==="
    done
fi
echo "=== All done $(date +%H:%M:%S) ==="
