#!/usr/bin/env bash
#
# Download the datasets EgoCITE reads.
#
#   egolife     lmms-lab/EgoLife      -> dataset/EgoLife       raw release: DenseCaption
#                                                              and Transcript SRTs plus
#                                                              video clips. Input to
#                                                              preprocessing.
#   egolifeqa   Ego-R1/Ego-R1-Data    -> dataset/Ego-R1-Data   EgoLifeQA questions
#   egor1       Ego-R1/Ego-R1-Bench   -> dataset/Ego-R1-Bench  Ego-R1-Bench questions
#                                                              (manual + gemini splits)
#
# Everything lands in EgoCITE/dataset/ so the project keeps its own copy. Point
# the pipeline at it afterwards:
#
#   export EGOCITE_DATASET=/path/to/EgoCITE/dataset
#
# Usage
# -----
#   bash src/scripts/0-download.sh [OPTIONS] [DATASET ...]
#
# Options
# -------
#   --dest DIR            Where to download (default: EgoCITE/dataset)
#   --include PATTERN     Only files matching this glob, e.g. 'EgoLifeCap/**'
#                         to take the SRTs and skip the video clips
#   --token TOKEN         HuggingFace token for gated repos
#                         (default: $HF_TOKEN, else `hf auth login`)
#   --max-workers N       Parallel file downloads. Unset by default (the hf
#                         CLI picks its own); pass a small number such as 2
#                         if the Hub rate-limits you.
#   --dry-run             Print what would be fetched, without downloading
#
# DATASET is egolife | egolifeqa | egor1; all three if none is named.
#
# Needs the `hf` CLI, which comes with huggingface_hub — install.sh installs it.
#
# Examples
# --------
#   bash src/scripts/0-download.sh
#   bash src/scripts/0-download.sh egolifeqa egor1
#   bash src/scripts/0-download.sh egolife --include 'EgoLifeCap/**'
#   bash src/scripts/0-download.sh --dry-run
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

DEST="$PROJECT_ROOT/dataset"
INCLUDE=""
TOKEN="${HF_TOKEN:-}"
MAX_WORKERS=""
DRY_RUN=""
DATASETS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dest)        DEST="$2";        shift 2 ;;
        --include)     INCLUDE="$2";     shift 2 ;;
        --token)       TOKEN="$2";       shift 2 ;;
        --max-workers) MAX_WORKERS="$2"; shift 2 ;;
        --dry-run)     DRY_RUN=1;        shift ;;
        -h|--help) sed -n '/^# Usage/,/^set -/p' "$0" | grep '^#' | sed 's/^# \?//'; exit 0 ;;
        -*) echo "Unknown option: $1" >&2; exit 1 ;;
        *)  DATASETS+=("$1"); shift ;;
    esac
done

[[ ${#DATASETS[@]} -eq 0 ]] && DATASETS=(egolife egolifeqa egor1)

repo_of() {
    case "$1" in
        egolife)   echo "lmms-lab/EgoLife" ;;
        egolifeqa) echo "Ego-R1/Ego-R1-Data" ;;
        egor1)     echo "Ego-R1/Ego-R1-Bench" ;;
        *) return 1 ;;
    esac
}
dir_of() {
    case "$1" in
        egolife)   echo "EgoLife" ;;
        egolifeqa) echo "Ego-R1-Data" ;;
        egor1)     echo "Ego-R1-Bench" ;;
        *) return 1 ;;
    esac
}

for D in "${DATASETS[@]}"; do
    repo_of "$D" >/dev/null || {
        echo "Unknown dataset '$D' (egolife | egolifeqa | egor1)" >&2; exit 1; }
done

HF=""
for C in hf huggingface-cli; do
    command -v "$C" >/dev/null 2>&1 && { HF="$C"; break; }
done
[[ -z "$HF" ]] && {
    echo "The HuggingFace CLI was not found. Install it with:" >&2
    echo "    pip install huggingface_hub        (or run install.sh)" >&2
    exit 1; }

echo "destination : $DEST"
echo "datasets    : ${DATASETS[*]}"
echo "cli         : $HF"
echo "token       : $([[ -n "$TOKEN" ]] && echo "yes" || echo "no (public repos only)")"
[[ -n "$INCLUDE" ]] && echo "include     : $INCLUDE"

FAILED=()

for D in "${DATASETS[@]}"; do
    REPO="$(repo_of "$D")"
    LOCAL="$DEST/$(dir_of "$D")"
    echo ""
    echo "========================================"
    echo "  $D : $REPO"
    echo "  -> $LOCAL"
    echo "========================================"

    CMD=("$HF" download "$REPO" --repo-type dataset --local-dir "$LOCAL")
    [[ -n "$MAX_WORKERS" ]] && CMD+=(--max-workers "$MAX_WORKERS")
    [[ -n "$INCLUDE" ]] && CMD+=(--include "$INCLUDE")
    [[ -n "$TOKEN"   ]] && CMD+=(--token "$TOKEN")
    echo "  ${CMD[*]//$TOKEN/\$TOKEN}"
    [[ -n "$DRY_RUN" ]] && continue

    mkdir -p "$LOCAL"
    "${CMD[@]}" || { echo "FAILED: $D ($REPO)" >&2; FAILED+=("$D"); }
done

echo ""
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "=== ${#FAILED[@]} failed: ${FAILED[*]} ==="
    echo "Gated repo? Accept the terms on its HuggingFace page, then pass --token"
    echo "or export HF_TOKEN."
    exit 1
fi
if [[ -z "$DRY_RUN" ]]; then
    echo "=== All done. Point the pipeline at it with:"
    echo "    export EGOCITE_DATASET=$DEST"
fi
