"""
build_conversation.py — segment captions into CONVERSATION TOPIC nodes.

The LLM is asked to identify only the conversation segments in each window and
label each one with the BROAD TOPIC being discussed. Captions with no active
conversation are simply not assigned to any group.

Pipeline (single script — no fix, no verify, no merge):
  1. Read raw.json (canonical captions).
  2. Build 30-min windows.
  3. Fire ALL windows as ONE OpenAI Batch API job (gpt-* default), or sequential
     real-time for non-batch models. reasoning_effort="none" for gpt-*.
  4. Parse each window's raw response → SegmentationOutput.
  5. Convert each group directly into a conversation node — NO embedding-merge,
     NO summary word-length enforcement (raw segmentation kept as-is).
  6. Write output/memory/{source}/{person}/captions/conversation.json
            output/memory/{source}/{person}/captions/conversation_window_log.json

Run after build_action.py:
    python episodic/build_action.py       --person A1_JAKE
    python episodic/build_conversation.py --person A1_JAKE
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from typing import List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logger = logging.getLogger("episodic.build_conversation")

# src/memory/<this file>  ->  src/  (so the flat `config` / `prompts` / `memory`
# modules resolve when this file is run directly as a script).
_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import config  # noqa: E402
import prompts  # noqa: E402

from memory.segmentation import (  # noqa: E402
    _CountWindowedSegmenter, SegmentationOutput, _segment_messages,
)
from models import make_llm  # noqa: E402


_SYSTEM_CONVERSATION = prompts.load("memory/conversation_segment")


class ConversationSegmenter(_CountWindowedSegmenter):
    """Captions → conversation-topic nodes.

    Same 30-minute windowing and merge logic as ActivitySegmenter; only the
    system prompt and label differ. No retry, no disentangle.
    """
    WINDOW_SEC        = 1800   # 30-minute non-overlapping windows
    _SYSTEM           = _SYSTEM_CONVERSATION
    _SOURCE_KEY       = "raw_caption_indices"
    _REQUIRE_OVERLAP  = True
    _ENABLE_THINKING  = False
    _LABEL            = "conversation"
    _SUMMARY_WORD_MIN = 10     # enforce conversation summary length to 10-20 words
    _SUMMARY_WORD_MAX = 20


def _parse_seg(raw: str) -> Optional[SegmentationOutput]:
    if not raw:
        return None
    try:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        return SegmentationOutput.model_validate_json(m.group() if m else raw)
    except Exception as exc:
        logger.debug("  [parse] %s", exc)
        return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Episodic conversation-topic segmentation.")
    p.add_argument("--person", required=True, help="Person ID, e.g. A1_JAKE")
    p.add_argument("--conv-llm-name", default="gpt-5.4-2026-03-05",
                   help="LLM for conversation segmentation. gpt-* uses the OpenAI "
                        "Batch API. Default: gpt-5.4-2026-03-05.")
    p.add_argument("--source", default="densecaption", choices=config.SOURCES,
                   help="Caption source: 'densecaption' uses the released EgoLife captions, "
                        "'gemini'/'gemma' use the VLM-generated ones.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    caps_dir = config.captions_dir(args.person, args.source, create=True)
    raw_file = os.path.join(caps_dir, "raw.json")
    conv_file   = os.path.join(caps_dir, "conversation.json")
    window_file = os.path.join(caps_dir, "conversation_window_log.json")

    if not os.path.exists(raw_file):
        logger.error("Not found: %s  (run build_action.py first)", raw_file)
        sys.exit(1)

    with open(raw_file) as f:
        raw_captions = json.load(f)
    logger.info("Person: %s (%d captions)", args.person, len(raw_captions))

    llm = make_llm(args.conv_llm_name)
    # Conversation topic extraction is a focused recognition task; "none" turns
    # off reasoning tokens entirely for gpt-* — cheapest and fastest.
    if hasattr(llm, "reasoning_effort"):
        llm.reasoning_effort = "none"
    use_batch = hasattr(llm, "generate_batch_api") and not getattr(llm, "_no_batch", False)
    logger.info("Conversation LLM: %s (%s, reasoning_effort=none)",
                args.conv_llm_name, "Batch API" if use_batch else "real-time")

    segmenter = ConversationSegmenter(llm_model=llm)
    windows = segmenter._build_windows(raw_captions)
    items_per_window = [[raw_captions[j] for j in idxs] for idxs in windows]
    prompts = [_segment_messages(segmenter._SYSTEM, items) for items in items_per_window]
    logger.info("Built %d windows (30-min each)", len(windows))

    # ----- 1. Batch LLM call -------------------------------------------
    t0 = time.time()
    if use_batch:
        raws = llm.generate_batch_api(
            prompts,
            enable_thinking=segmenter._ENABLE_THINKING,
            description="conversation",
        )
    else:
        import concurrent.futures as _cf
        with _cf.ThreadPoolExecutor(max_workers=8) as _ex:
            raws = list(_ex.map(
                lambda p: llm.generate(p, enable_thinking=segmenter._ENABLE_THINKING),
                prompts))
    logger.info("LLM phase done in %.1fs", time.time() - t0)

    # ----- 2. Parse responses ------------------------------------------
    parsed: List[Optional[SegmentationOutput]] = [_parse_seg(r) for r in raws]
    n_with_groups = sum(1 for r in parsed if r is not None and r.groups)
    logger.info("Parse: %d/%d windows have at least one conversation group",
                n_with_groups, len(parsed))

    # ----- 3. Assemble nodes directly (no merge phase) -----------------
    # Each group becomes one conversation node. No embedding-similarity merge,
    # no word-length enforcement — caller asked for the raw segmentation.
    from memory.segmentation import _time_to_sec, _date_to_day  # noqa: E402

    conversation_nodes: List[dict] = []
    window_log: List[dict] = []
    for wi, (idxs, items, result) in enumerate(zip(windows, items_per_window, parsed)):
        win_entry = {
            "window_idx": wi,
            "input_range": {
                "start": f"{items[0]['date']} {items[0]['start_time']}",
                "end":   f"{items[-1]['date']} {items[-1]['end_time']}",
            },
            "input_count": len(items),
            "fallback":    result is None,
            "groups":      [],
        }
        if result is not None:
            for group in result.groups:
                local = [j for j in group.indices if 0 <= j < len(items)]
                if not local:
                    continue
                win_entry["groups"].append(
                    {"indices": group.indices, "summary": group.summary})
                conversation_nodes.append(segmenter._make_node(
                    [items[j] for j in local],
                    [idxs[j] for j in local],
                    group.summary,
                ))
        window_log.append(win_entry)

    # Drop invalid time ranges, sort by (date, start_time).
    conversation_nodes = [
        n for n in conversation_nodes
        if _time_to_sec(n["start_time"]) <= _time_to_sec(n["end_time"])
    ]
    conversation_nodes.sort(key=lambda n: (_date_to_day(n["date"]),
                                            _time_to_sec(n["start_time"])))

    # ----- 4. Write ----------------------------------------------------
    with open(conv_file, "w") as f:
        json.dump(conversation_nodes, f, indent=2, ensure_ascii=False)
    with open(window_file, "w") as f:
        json.dump(window_log, f, indent=2, ensure_ascii=False)
    logger.info("Wrote %d conversation nodes → %s",
                len(conversation_nodes), conv_file)


if __name__ == "__main__":
    main()
