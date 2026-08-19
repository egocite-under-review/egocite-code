"""
build_activity.py — captions → activity nodes (the whole activity stage).

Four phases, all in this one script:

  1. SEGMENT      30-minute windows over the canonical captions (raw.json); every
                  window prompt goes out as ONE OpenAI Batch API job (or
                  concurrent real-time calls for non-batch models). The raw
                  per-window responses are saved to activity_raw.json so the
                  later phases can be re-run without paying for the LLM again
                  (--skip-segmentation).
  2. RETRY        Re-prompt any window whose response is unparseable, has no
                  groups, or contains a block longer than _MAX_BLOCK_SEC (10 min),
                  with a stricter system prompt — batched, up to _MAX_RETRIES rounds.
  3. DISENTANGLE  Split every compound summary (one matching /and|while|then/)
                  into one summary per action — one batched LLM job.
  4. MERGE        Sequential cross-window merge: embedding-similarity dedup,
                  LLM-merge of recurring same-day blocks, and enforcement of the
                  10–20 word summary length.

Writes to output/memory/{source}/{person}/captions/:
    activity_raw.json          raw per-window LLM responses (provenance / resume)
    activity.json              the activity nodes
    activity_window_log.json   per-window record of what was produced and merged

Run after build_action.py and before build.py:
    python src/memory/build_action.py    --person A1_JAKE
    python src/memory/build_activity.py  --person A1_JAKE
    python src/memory/build.py           --person A1_JAKE
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import Counter
from typing import List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logger = logging.getLogger("memory.build_activity")

# src/memory/<this file>  ->  src/  (so the flat `config` / `prompts` / `memory`
# modules resolve when this file is run directly as a script).
_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import config  # noqa: E402
import prompts  # noqa: E402

from memory.segmentation import (  # noqa: E402
    ActivitySegmenter, SegmentationOutput, Group,
    _segment_messages, abs_sec,
    _COMPOUND_RE, _DISENTANGLE_SYSTEM,
)
from models import make_llm  # noqa: E402


def _use_batch(llm) -> bool:
    """True when the backend supports the OpenAI Batch API (the LiteLLM proxy
    streams instead and is tagged `_no_batch`)."""
    return hasattr(llm, "generate_batch_api") and not getattr(llm, "_no_batch", False)


def _parse_seg(raw: str) -> Optional[SegmentationOutput]:
    if not raw:
        return None
    try:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        return SegmentationOutput.model_validate_json(m.group() if m else raw)
    except Exception as exc:
        logger.debug("  [parse] %s", exc)
        return None


def _max_block_seconds(result: SegmentationOutput, items: List[dict]) -> float:
    worst = 0.0
    for g in result.groups:
        local = [j for j in g.indices if 0 <= j < len(items)]
        if not local:
            continue
        s = abs_sec(items[min(local)]["date"], items[min(local)]["start_time"])
        e = abs_sec(items[max(local)]["date"], items[max(local)]["end_time"])
        worst = max(worst, e - s)
    return worst


# ---------------------------------------------------------------------------
# Phase 1 — window segmentation
# ---------------------------------------------------------------------------

def _run_segmentation(llm, segmenter, args, raw_captions: List[dict],
                      windows: List[List[int]], items_per_window: List[List[dict]],
                      out_file: str) -> dict:
    """Fire one segmentation prompt per window and record the raw responses."""
    msgs = [_segment_messages(segmenter._SYSTEM, items) for items in items_per_window]
    logger.info("Segmentation: %d windows (%s)", len(windows),
                "Batch API" if _use_batch(llm) else "real-time")

    t0 = time.time()
    if _use_batch(llm):
        raws = llm.generate_batch_api(
            msgs, enable_thinking=segmenter._ENABLE_THINKING, description="activity")
    else:
        import concurrent.futures as _cf
        with _cf.ThreadPoolExecutor(max_workers=8) as _ex:
            raws = list(_ex.map(
                lambda p: llm.generate(p, enable_thinking=segmenter._ENABLE_THINKING),
                msgs))

    batch = {
        "person":        args.person,
        "activity_llm":  args.activity_llm_name,
        "system_prompt": segmenter._SYSTEM,
        "n_windows":     len(windows),
        "windows": [
            {
                "window_idx":     wi,
                "global_indices": list(idxs),
                "input_range": {
                    "start": f"{items[0]['date']} {items[0]['start_time']}",
                    "end":   f"{items[-1]['date']} {items[-1]['end_time']}",
                },
                "input_count":  len(items),
                "raw_response": raws[wi] or "",
            }
            for wi, (idxs, items) in enumerate(zip(windows, items_per_window))
        ],
    }
    with open(out_file, "w") as f:
        json.dump(batch, f, indent=2, ensure_ascii=False)
    logger.info("Segmentation done in %.1fs → %s", time.time() - t0, out_file)
    return batch


# ---------------------------------------------------------------------------
# Phase 2 — retry the bad windows
# ---------------------------------------------------------------------------

def _retry_reason(result: Optional[SegmentationOutput], items: List[dict],
                  max_block_sec: int) -> Optional[str]:
    """Why this window still needs a re-prompt, or None if it's fine."""
    if result is None:
        return "no_parse"
    if not result.groups:
        return "empty_groups"
    worst = _max_block_seconds(result, items)
    if worst > max_block_sec:
        return f"over_long({worst / 60:.1f}min)"
    return None


def _redo_system(reason: str, base_system: str, max_block_sec: int) -> str:
    """Build the stricter system prompt for a retry, tailored to the failure mode."""
    if reason == "no_parse":
        note = prompts.load("memory/retry_no_parse")
    elif reason == "empty_groups":
        note = prompts.load("memory/retry_empty_groups")
    else:                                  # over_long(...)
        note = prompts.render("memory/retry_over_long", max_min=max_block_sec // 60)
    return base_system + "\n\n" + note


def _batched_retry(llm, items_per_window: List[List[dict]],
                   parsed: List[Optional[SegmentationOutput]], *,
                   system: str, max_block_sec: int, max_retries: int,
                   enable_thinking: bool) -> None:
    """Re-prompt bad windows in BATCHES across up to `max_retries` rounds.

    Each round: collect every window that's still bad (no parse / empty groups /
    over-long block), build per-window redo prompts, fire ONE Batch API job for
    that round, parse the responses back into `parsed`. Stops early when no
    windows are bad. Mutates `parsed` in place.
    """
    for attempt in range(1, max_retries + 1):
        bad: List[tuple] = []  # (wi, reason)
        for wi, (result, items) in enumerate(zip(parsed, items_per_window)):
            reason = _retry_reason(result, items, max_block_sec)
            if reason is not None:
                bad.append((wi, reason))
        if not bad:
            logger.info("Retry round %d: no bad windows — stopping", attempt)
            return

        by_reason = Counter(r.split("(")[0] for _, r in bad)
        logger.info("Retry round %d/%d: %d windows need re-prompt (%s)",
                    attempt, max_retries, len(bad),
                    ", ".join(f"{k}={v}" for k, v in by_reason.most_common()))

        msgs = [
            _segment_messages(_redo_system(reason, system, max_block_sec),
                              items_per_window[wi])
            for wi, reason in bad
        ]

        t0 = time.time()
        if _use_batch(llm):
            raws = llm.generate_batch_api(
                msgs, enable_thinking=enable_thinking, description=f"retry-{attempt}")
        else:
            raws = [llm.generate(p, enable_thinking=enable_thinking) for p in msgs]
        logger.info("Retry round %d: batch returned %d responses in %.1fs",
                    attempt, len(raws), time.time() - t0)

        for (wi, _), raw in zip(bad, raws):
            parsed[wi] = _parse_seg(raw)

    # Final summary of remaining bad windows after all rounds.
    still_bad: List[tuple] = []
    for wi, (result, items) in enumerate(zip(parsed, items_per_window)):
        reason = _retry_reason(result, items, max_block_sec)
        if reason is not None:
            still_bad.append((wi, reason))
    if still_bad:
        by_reason = Counter(r.split("(")[0] for _, r in still_bad)
        logger.warning("After %d retry rounds, %d windows still bad (%s)",
                       max_retries, len(still_bad),
                       ", ".join(f"{k}={v}" for k, v in by_reason.most_common()))


# ---------------------------------------------------------------------------
# Phase 3 — disentangle compound summaries
# ---------------------------------------------------------------------------

def _batched_disentangle(llm, parsed: List[Optional[SegmentationOutput]]) -> int:
    """Split every compound-summary group across all windows in ONE Batch API job.

    A 'compound' group is one whose summary matches /and|while|then/. Each such
    group's summary is sent to the disentangle LLM and the response is parsed
    into one or more new summaries; the new Groups inherit the original indices.
    Non-compound groups are left untouched. Returns the number of LLM calls made
    (i.e. compound groups processed).
    """
    # Collect every compound group across all windows.
    targets: List[tuple] = []  # (window_idx, group_idx, summary)
    for wi, result in enumerate(parsed):
        if result is None or not result.groups:
            continue
        for gi, g in enumerate(result.groups):
            if _COMPOUND_RE.search(g.summary):
                targets.append((wi, gi, g.summary))
    if not targets:
        logger.info("Disentangle: no compound summaries found — nothing to split")
        return 0

    logger.info("Disentangle: %d compound groups across %d windows — batching",
                len(targets), sum(1 for r in parsed if r is not None and r.groups))

    msgs = [
        [{"role": "system", "content": _DISENTANGLE_SYSTEM},
         {"role": "user",   "content": summary}]
        for _, _, summary in targets
    ]
    if _use_batch(llm):
        raws = llm.generate_batch_api(msgs, enable_thinking=False,
                                      description="disentangle")
    else:
        raws = [llm.generate(p, enable_thinking=False) for p in msgs]

    # Index per-window groups by ORIGINAL position so we can replace them.
    replacements: dict = {}  # (wi, gi) → List[Group]
    n_ok = 0
    for (wi, gi, summary), raw in zip(targets, raws):
        try:
            m = re.search(r"\{.*\}", raw or "", re.DOTALL)
            data = json.loads(m.group() if m else (raw or ""))
            acts = [str(s).strip() for s in data.get("activities", []) if str(s).strip()]
            if not acts:
                raise ValueError("empty activities list")
            orig = parsed[wi].groups[gi]
            replacements[(wi, gi)] = [Group(indices=orig.indices, summary=s) for s in acts]
            n_ok += 1
        except Exception as exc:
            logger.warning("  [disentangle parse failed] %r: %s — keeping original",
                           summary, exc)

    # Rebuild each window's group list, expanding compound groups in place.
    for wi, result in enumerate(parsed):
        if result is None or not result.groups:
            continue
        new_groups: List[Group] = []
        for gi, g in enumerate(result.groups):
            new_groups.extend(replacements.get((wi, gi), [g]))
        result.groups = new_groups

    logger.info("Disentangle: %d/%d groups split successfully", n_ok, len(targets))
    return len(targets)


# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Episodic activity stage: segment → retry → disentangle → merge.")
    p.add_argument("--person", required=True, help="Person ID, e.g. A1_JAKE")
    p.add_argument("--activity-llm-name", default="gpt-5.4-2026-03-05",
                   help="LLM for every phase. gpt-* uses the OpenAI Batch API. "
                        "Default: gpt-5.4-2026-03-05.")
    p.add_argument("--source", default="densecaption", choices=config.SOURCES,
                   help="Caption source: 'densecaption' uses the released EgoLife captions, "
                        "'gemini'/'gemma' use the VLM-generated ones.")
    p.add_argument("--skip-segmentation", action="store_true",
                   help="Reuse the existing activity_raw.json instead of re-running "
                        "the window segmentation (phases 2-4 only).")
    p.add_argument("--skip-retry", action="store_true",
                   help="Skip the retry pass (still parse + disentangle + merge).")
    p.add_argument("--skip-disentangle", action="store_true",
                   help="Skip the disentangle pass.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    caps_dir      = config.captions_dir(args.person, args.source, create=True)
    raw_file      = os.path.join(caps_dir, "raw.json")
    seg_file      = os.path.join(caps_dir, "activity_raw.json")
    activity_file = os.path.join(caps_dir, "activity.json")
    window_file   = os.path.join(caps_dir, "activity_window_log.json")

    if not os.path.exists(raw_file):
        logger.error("Not found: %s  (run build_action.py first)", raw_file)
        sys.exit(1)
    with open(raw_file) as f:
        raw_captions = json.load(f)
    logger.info("Person: %s (%d captions)", args.person, len(raw_captions))

    llm = make_llm(args.activity_llm_name)
    logger.info("Activity LLM: %s", args.activity_llm_name)
    segmenter = ActivitySegmenter(llm_model=llm)

    windows = segmenter._build_windows(raw_captions)
    items_per_window = [[raw_captions[j] for j in idxs] for idxs in windows]
    logger.info("Built %d windows (30-min each)", len(windows))

    # ----- 1. Window segmentation --------------------------------------
    if args.skip_segmentation:
        if not os.path.exists(seg_file):
            logger.error("Not found: %s  (drop --skip-segmentation)", seg_file)
            sys.exit(1)
        with open(seg_file) as f:
            batch = json.load(f)
        logger.info("Reusing %d window responses from %s",
                    len(batch["windows"]), seg_file)
        items_per_window = [
            [raw_captions[j] for j in w["global_indices"]] for w in batch["windows"]
        ]
    else:
        batch = _run_segmentation(llm, segmenter, args, raw_captions,
                                  windows, items_per_window, seg_file)

    parsed: List[Optional[SegmentationOutput]] = [
        _parse_seg(w.get("raw_response", "")) for w in batch["windows"]
    ]
    logger.info("Parse: %d/%d windows parseable with non-empty groups",
                sum(1 for r in parsed if r is not None and r.groups), len(parsed))

    # Retry / disentangle / merge are well-defined short tasks — "low" reasoning
    # effort is plenty on gpt-* and far cheaper than the segmentation default.
    if hasattr(llm, "reasoning_effort"):
        llm.reasoning_effort = "low"

    # ----- 2. Retry the bad / over-long windows ------------------------
    if not args.skip_retry:
        t = time.time()
        _batched_retry(
            llm, items_per_window, parsed,
            system          = segmenter._SYSTEM,
            max_block_sec   = segmenter._MAX_BLOCK_SEC,
            max_retries     = segmenter._MAX_RETRIES,
            enable_thinking = segmenter._ENABLE_THINKING,
        )
        logger.info("Retry phase done in %.1fs", time.time() - t)

    # ----- 3. Disentangle compound summaries ---------------------------
    if not args.skip_disentangle:
        t = time.time()
        n_done = _batched_disentangle(llm, parsed)
        logger.info("Disentangle done: %d groups in %.1fs", n_done, time.time() - t)

    # ----- 4. Merge across windows -------------------------------------
    # Inject the parsed results so segment() skips its own LLM phase and only
    # runs the embedding-similarity merge + word-length enforcement.
    segmenter._segment_windows = lambda items_per_window: parsed  # noqa: E731

    t = time.time()
    activity_nodes = segmenter.segment(raw_captions)
    logger.info("Merge done in %.1fs → %d activity nodes",
                time.time() - t, len(activity_nodes))

    with open(activity_file, "w") as f:
        json.dump(activity_nodes, f, indent=2, ensure_ascii=False)
    with open(window_file, "w") as f:
        json.dump(segmenter.window_log, f, indent=2, ensure_ascii=False)
    logger.info("Wrote %d activity nodes → %s", len(activity_nodes), activity_file)


if __name__ == "__main__":
    main()
