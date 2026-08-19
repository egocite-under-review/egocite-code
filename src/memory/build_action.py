"""
build_action.py — Stage 0 of the episodic build: action extraction.

Reads the raw 30-sec captions and extracts atomic action entries from each
(one LLM request per caption, with the preceding --action-context-size captions
as context for reference resolution). Writes:
  output/memory/{source}/{person}/captions/raw.json     (the canonical caption list)
  output/memory/{source}/{person}/captions/action.json  (the extracted action nodes)

A gpt-* model runs via the OpenAI Batch API (50% cheaper); local models fire
concurrent real-time requests. Run before build.py (and optionally fix_action.py
to re-extract any missing captions):

    python episodic/build_action.py --person A1_JAKE --action-llm-name gpt-5.4-mini-2026-03-17
    python episodic/build.py        --person A1_JAKE
"""

import argparse
import glob
import json
import logging
import os
import sys
import time

from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logger = logging.getLogger("episodic.build_action")

# src/memory/<this file>  ->  src/  (so the flat `config` / `prompts` / `memory`
# modules resolve when this file is run directly as a script).
_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import config  # noqa: E402

from memory.segmentation import ActionExtractor  # noqa: E402
from models import make_llm  # noqa: E402

# Concurrency for the local/real-time path (I/O-bound LLM calls).
_CONCURRENCY = 8


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Episodic action extraction (Stage 0).")
    p.add_argument("--person", required=True, help="Person ID, e.g. A1_JAKE")
    p.add_argument("--action-llm-name", default="Qwen/Qwen3.6-27B-FP8",
                   help="LLM for action extraction. gpt-* uses the Batch API. "
                        "Default: Qwen/Qwen3.6-27B-FP8.")
    p.add_argument("--action-context-size", type=int, default=10,
                   help="Preceding captions given as context for reference resolution. Default: 10.")
    p.add_argument("--until-day", type=int, default=None,
                   help="Only process captions up to and including this day.")
    p.add_argument("--source", default="densecaption", choices=config.SOURCES,
                   help="Caption source: 'densecaption' (default) reads the released "
                        "EgoLife captions, 'gemini'/'gemma' read the VLM-generated ones. Both "
                        "write to output/memory/{source}/{person}/.")
    p.add_argument("--limit", type=int, default=None,
                   help="Process at most N captions (for testing).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    person = args.person

    # INPUT: the per-person 30-sec caption list (densecaption or VLM-generated).
    # OUTPUT: output/memory/{source}/{person}/captions/
    caps_dir    = config.captions_dir(person, args.source, create=True)
    raw_file    = os.path.join(caps_dir, "raw.json")
    action_file = os.path.join(caps_dir, "action.json")

    motion_file = config.raw_caption_file(person, args.source)
    if not os.path.exists(motion_file):
        logger.error("Caption file not found: %s", motion_file)
        sys.exit(1)
    with open(motion_file) as f:
        raw_captions = json.load(f)
    logger.info("Loaded %d %s captions from %s", len(raw_captions), args.source, motion_file)
    if args.limit is not None:
        raw_captions = raw_captions[:args.limit]
        logger.info("Limited to %d captions", len(raw_captions))

    if args.until_day is not None:
        before = len(raw_captions)
        raw_captions = [c for c in raw_captions
                        if int(c["date"].replace("DAY", "")) <= args.until_day]
        logger.info("Person: %s (DAY1–DAY%d, %d/%d captions)",
                    person, args.until_day, len(raw_captions), before)
    else:
        logger.info("Person: %s (%d captions)", person, len(raw_captions))

    # raw.json is the canonical caption list that downstream stages index into.
    with open(raw_file, "w") as f:
        json.dump(raw_captions, f, indent=2, ensure_ascii=False)

    action_llm = make_llm(args.action_llm_name)
    logger.info("Action LLM: %s", args.action_llm_name)
    extractor = ActionExtractor(llm_model=action_llm)
    ctx_n = max(0, args.action_context_size)
    t0 = time.time()

    def _prompt(idx):
        context = raw_captions[max(0, idx - ctx_n):idx]
        return extractor.draft_messages(raw_captions[idx]["text"], context_captions=context)

    def _to_nodes(idx, raw):
        actions, _ = extractor.parse_draft(raw)
        cap = raw_captions[idx]
        logger.info("  [action] %s %s-%s  %s",
                    cap["date"], cap["start_time"], cap["end_time"],
                    " | ".join(actions) if actions else "(none)")
        return extractor.to_nodes(cap, idx, actions)

    use_batch = hasattr(action_llm, "generate_batch_api") and not getattr(action_llm, "_no_batch", False)
    if use_batch:
        logger.info("Action extraction: %d captions via OpenAI Batch API (context=prev %d)",
                    len(raw_captions), ctx_n)
        prompts = [_prompt(i) for i in range(len(raw_captions))]
        try:
            raws = action_llm.generate_batch_api(
                prompts, enable_thinking=False, description="action extraction")
            action_nodes = [n for i, r in enumerate(raws) for n in _to_nodes(i, r)]
        except Exception as exc:
            logger.warning("Batch API failed (%s) — falling back to concurrent real-time calls", exc)
            use_batch = False

    if not use_batch:
        logger.info("Action extraction: %d captions, concurrent requests (context=prev %d)",
                    len(raw_captions), ctx_n)
        from pqdm.threads import pqdm as _pqdm

        def _extract(idx):
            try:
                raw = action_llm.generate(_prompt(idx), enable_thinking=False)
            except Exception as exc:
                logger.warning("  [action] call error for caption %d: %s", idx, exc)
                raw = ""
            return _to_nodes(idx, raw)

        results = _pqdm(list(range(len(raw_captions))), _extract,
                        n_jobs=_CONCURRENCY, desc="Extracting actions",
                        unit="cap", tqdm_class=tqdm)
        action_nodes = [node for batch in results if batch for node in batch]

    with open(action_file, "w") as f:
        json.dump(action_nodes, f, indent=2, ensure_ascii=False)
    logger.info("Extracted %d action nodes in %.1fs → %s",
                len(action_nodes), time.time() - t0, action_file)


if __name__ == "__main__":
    main()
