"""
LLM-based segmentation primitives shared by the memory-build stages.

  ActionExtractor          raw 30-sec caption -> atomic action nodes
  ActivitySegmenter        raw 30-sec captions -> activity nodes (30-min windows,
                           multiple contiguous 1-10 min activities per window)
  _CountWindowedSegmenter  the windowing / merge machinery both the activity and
                           the conversation segmenter build on

All boundary decisions are made by the LLM. Blocks that continue across a window
boundary — or recur later the same day — are merged by embedding similarity.
Prompts live in prompt/memory/*.txt and are loaded through `prompts`.
"""

import json
import logging
import re
from collections import defaultdict
from typing import List, Optional

import numpy as np

from pydantic import BaseModel, field_validator, model_validator
from pydantic import ConfigDict
from tqdm import tqdm
import prompts


logger = logging.getLogger(__name__)

_COMPOUND_RE = re.compile(r'\b(and|while|then)\b', re.IGNORECASE)

_MERGE_SYSTEM = prompts.load("memory/merge_summaries")

_DEDUP_EMBED_MODEL = "Qwen/Qwen3-Embedding-4B"
_DEDUP_SIM_THRESHOLD = 0.9


def _merge_or_append(
    llm, em, segments: List[dict], vecs: List[np.ndarray], new_seg: dict, *,
    index_keys: List[str], require_overlap: bool,
    sim_threshold: float = _DEDUP_SIM_THRESHOLD, scan_recent: int = 40,
) -> None:
    """
    Append `new_seg` to `segments`, OR merge it into the best-matching recent
    SAME-DAY segment when cosine similarity > `sim_threshold` (LLM-merged).

    require_overlap=True  → only merge segments whose time spans overlap
                            (a block continued across a window boundary).
    require_overlap=False → merge any sufficiently-similar same-day segment
                            (a recurring intention split across the day).

    `index_keys` lists the dict keys whose integer-index lists are unioned on a
    merge. `vecs` is kept in sync with `segments`.
    """
    new_start = _time_to_sec(new_seg["start_time"])
    new_end   = _time_to_sec(new_seg["end_time"])
    if new_start > new_end:
        return

    new_vec = em.encode([new_seg["text"]], batch_size=1)[0]
    new_vec = new_vec / max(float(np.linalg.norm(new_vec)), 1e-9)

    best_idx, best_sim = -1, sim_threshold
    lo = max(0, len(segments) - scan_recent)
    for idx in range(len(segments) - 1, lo - 1, -1):
        prev = segments[idx]
        if prev["date"] != new_seg["date"]:
            break  # segments are time-ordered → a different date means past the day
        if require_overlap:
            p_start = _time_to_sec(prev["start_time"])
            p_end   = _time_to_sec(prev["end_time"])
            if not (p_end >= new_start and new_end >= p_start):
                continue
        sim = float(vecs[idx] @ new_vec)
        if sim > best_sim:
            best_sim, best_idx = sim, idx

    if best_idx >= 0:
        prev = segments[best_idx]
        try:
            raw = llm.generate(
                [{"role": "system", "content": _MERGE_SYSTEM},
                 {"role": "user", "content":
                     f"Summary 1: {prev['text']!r}\nSummary 2: {new_seg['text']!r}"}],
                enable_thinking=False,
            )
            merged_text = json.loads(
                re.search(r"\{.*\}", raw, re.DOTALL).group()
            )["summary"].strip()
            if not merged_text:
                raise ValueError("empty merged summary")
            if new_start < _time_to_sec(prev["start_time"]):
                prev["start_time"] = new_seg["start_time"]
            if new_end > _time_to_sec(prev["end_time"]):
                prev["end_time"] = new_seg["end_time"]
            prev["text"] = merged_text
            for k in index_keys:
                prev[k] = sorted(set(prev.get(k, []) + new_seg.get(k, [])))
            mv = em.encode([merged_text], batch_size=1)[0]
            vecs[best_idx] = mv / max(float(np.linalg.norm(mv)), 1e-9)
            logger.debug("  [dedup] sim=%.3f merged → %r", best_sim, merged_text)
            return
        except Exception as e:
            logger.debug("  [dedup merge failed] %r + %r: %s",
                         prev["text"], new_seg["text"], e)

    segments.append(new_seg)
    vecs.append(new_vec)


_DISENTANGLE_SYSTEM = prompts.load("memory/disentangle_summary")


def _enforce_word_range(llm, text: str, *, lo: int, hi: int,
                        context: str = "", retries: int = 5) -> str:
    """
    Rewrite `text` via the LLM until it has lo..hi words. Every fix is an LLM
    rewrite — NEVER a truncation. If no attempt lands in range within `retries`
    calls, the LLM result closest to the range is kept.

    `context` (e.g. the source captions) is shown to the LLM so it can add or
    trim TRUE specific detail rather than padding with filler.
    """
    if lo <= len(text.split()) <= hi:
        return text
    system = prompts.render("memory/rewrite_word_range", lo=lo, hi=hi)

    def _dist(s: str) -> int:
        wc = len(s.split())
        return 0 if lo <= wc <= hi else min(abs(wc - lo), abs(wc - hi))

    def _user(summary: str) -> str:
        if context:
            return f"CONTEXT (source captions):\n{context}\n\nSUMMARY TO REWRITE:\n{summary}"
        return summary

    best = text
    for _ in range(retries):
        try:
            raw = llm.generate(
                [{"role": "system", "content": system},
                 {"role": "user",   "content": _user(best)}],
                enable_thinking=False,
            )
            new = json.loads(re.search(r"\{.*\}", raw, re.DOTALL).group())["summary"].strip()
        except Exception as e:
            logger.debug("  [length-fix failed] %r: %s", best, e)
            new = ""
        if not new:
            continue
        if lo <= len(new.split()) <= hi:
            return new
        if _dist(new) < _dist(best):
            best = new
    return best


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

def _time_to_sec(time_str: str) -> int:
    t = time_str.zfill(8)
    return int(t[0:2]) * 3600 + int(t[2:4]) * 60 + int(t[4:6])


def _date_to_day(date_str: str) -> int:
    return int(date_str.replace("DAY", "")) - 1


def abs_sec(date: str, time_str: str) -> int:
    return _date_to_day(date) * 86400 + _time_to_sec(time_str)


def _dur_min(start: str, end: str) -> float:
    return (_time_to_sec(end) - _time_to_sec(start)) / 60.0


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class ActionList(BaseModel):
    actions: List[str]

    @field_validator("actions")
    @classmethod
    def non_empty_strings(cls, v):
        return [s.strip() for s in v if isinstance(s, str) and s.strip()]


class Group(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    indices: List[int] = []
    summary: str = ""

    @model_validator(mode="before")
    @classmethod
    def normalise_fields(cls, data):
        if isinstance(data, dict):
            # accept caption_indices as alias for indices
            if "indices" not in data and "caption_indices" in data:
                data["indices"] = data.pop("caption_indices")
            # coerce any stringified ints
            raw = data.get("indices", [])
            data["indices"] = [int(i) for i in raw if str(i).lstrip("-").isdigit()]
        return data


class SegmentationOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")
    groups: List[Group] = []


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM_ACTION_EXTRACT = prompts.load("memory/action_extract")


_SYSTEM_ACTIVITY = prompts.load("memory/activity_segment")
_ACTION_WORD_MAX = 20


def _truncate_words(s: str, limit: int) -> str:
    words = s.split()
    return " ".join(words[:limit]) if len(words) > limit else s


class ActionExtractor:
    """Extracts atomic action phrases from a single 30-sec caption.

    Stateless apart from the LLM handle: build the messages with
    `draft_messages`, parse the reply with `parse_draft`, and turn the result
    into action nodes with `to_nodes`. A caption whose extraction fails never
    disappears — it falls back to its own truncated text.
    """

    def __init__(self, llm_model) -> None:
        self.llm = llm_model

    @staticmethod
    def draft_messages(caption_text: str,
                       context_captions: Optional[List[dict]] = None) -> list:
        """The chat messages for a single-caption draft extraction.

        `context_captions` — the preceding captions, shown for reference resolution
        only (the model is told NOT to extract from them). When given, the user
        message gets a PRECEDING CAPTIONS block before the caption to process.
        """
        if not context_captions:
            return [{"role": "system", "content": _SYSTEM_ACTION_EXTRACT},
                    {"role": "user",   "content": caption_text}]
        ctx = "\n".join(
            f"({c['date']} {c['start_time']}-{c['end_time']}) {c['text']}"
            for c in context_captions
        )
        user = (
            "PRECEDING CAPTIONS (context only — do NOT extract from these; use them "
            "ONLY to resolve pronouns/references):\n" + ctx +
            "\n\nCAPTION TO PROCESS (extract actions from THIS one only):\n" + caption_text
        )
        return [{"role": "system", "content": _SYSTEM_ACTION_EXTRACT},
                {"role": "user",   "content": user}]

    @staticmethod
    def parse_draft(raw: str) -> tuple:
        """Parse a draft extraction's raw output → (actions, raw_text):
          - actions:  parsed action list, or [] if the output failed JSON parsing;
          - raw_text: the raw LLM output, handed to the verifier when parsing failed.
        """
        if not raw:
            return [], ""
        try:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            parsed = ActionList.model_validate_json(match.group() if match else raw)
            return parsed.actions, raw
        except Exception as exc:
            logger.debug("  [LLM extract] parse error: %s — passing raw output to verifier", exc)
            return [], raw

    @staticmethod
    def to_nodes(caption: dict, caption_idx: int, actions: List[str]) -> List[dict]:
        """Assemble action dicts for a caption (with the never-empty fallback)."""
        if not actions:
            actions = [_truncate_words(caption["text"], _ACTION_WORD_MAX)]
        return [
            {
                "text":                act,
                "start_time":          caption["start_time"],
                "end_time":            caption["end_time"],
                "date":                caption["date"],
                "raw_caption_indices": [caption_idx],
            }
            for act in actions
        ]


# ---------------------------------------------------------------------------
# Shared segmentation helpers
# ---------------------------------------------------------------------------

def _segment_messages(system_prompt: str, items: List[dict]) -> list:
    """Chat messages for one window's segmentation (shared by real-time + batch)."""
    items_text = "\n".join(
        f"[{i}] ({c['date']} {c['start_time']}-{c['end_time']}) {c['text']}"
        for i, c in enumerate(items)
    )
    user_content = f"Items to segment (index 0 to {len(items) - 1}):\n{items_text}"
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_content},
    ]


# ---------------------------------------------------------------------------
# Concrete segmenters
# ---------------------------------------------------------------------------

class _CountWindowedSegmenter:
    """
    Segments a flat, time-ordered list of items into higher-level blocks.

    Windowing — each LLM call sees one window of items (no item is ever dropped):
      * WINDOW_SEC set    → fixed-width time windows (e.g. 10-minute buckets)
      * WINDOW_SEC None   → fixed-count windows of WINDOW_ITEMS items
    A block that spans a window boundary, or a recurring same-day block, is merged
    into an earlier block by embedding similarity (LLM-merged on a hit).

    Output node schema (one per produced block):
      text, date, start_time, end_time, raw_caption_indices, <_SOURCE_KEY>
    where <_SOURCE_KEY> holds global indices into the `inputs` list — the link
    back to the layer below. An input lacking `raw_caption_indices` (e.g. a raw
    caption) contributes its own global index instead.
    """
    WINDOW_ITEMS: int = 100
    WINDOW_SEC: Optional[int] = None  # set → time-windowing instead of count-windowing
    _SYSTEM: str = ""
    _SOURCE_KEY: str = ""             # node key holding global indices into `inputs`
    _REQUIRE_OVERLAP: bool = True     # True: only merge time-overlapping blocks
    _ENABLE_THINKING: bool = False    # thinking mode for the segmentation LLM call
    _LABEL: str = "segment"
    _SUMMARY_WORD_MIN: Optional[int] = None  # set BOTH → enforce summary word count
    _SUMMARY_WORD_MAX: Optional[int] = None

    def __init__(self, llm_model) -> None:
        self.llm = llm_model
        self.window_log: List[dict] = []

    def _segment_windows(self,
                         items_per_window: List[List[dict]]
                         ) -> List[Optional[SegmentationOutput]]:
        """Return one parsed `SegmentationOutput` (or None) per window.

        The build stages run the LLM themselves — one Batch API job for all
        windows, with their own retry pass — and inject the parsed results by
        assigning to this attribute before calling `segment()`
        (see build_activity.py). `segment()` then only does the merge phase."""
        raise NotImplementedError(
            "assign _segment_windows with the parsed per-window results before "
            "calling segment()")

    def _build_windows(self, inputs: List[dict]) -> List[List[int]]:
        """Return a list of windows, each a list of global indices into `inputs`.

        WINDOW_SEC None → fixed-count windows of WINDOW_ITEMS items.
        WINDOW_SEC set  → fixed-width, clock-aligned, non-overlapping time windows."""
        n = len(inputs)
        if not self.WINDOW_SEC:
            return [list(range(i, min(i + self.WINDOW_ITEMS, n)))
                    for i in range(0, n, self.WINDOW_ITEMS)]
        buckets: dict = defaultdict(list)
        for idx, c in enumerate(inputs):
            b = abs_sec(c["date"], c["start_time"]) // self.WINDOW_SEC
            buckets[b].append(idx)
        return [buckets[b] for b in sorted(buckets)]

    def _make_node(self, group_items: List[dict], global_idxs: List[int], summary: str) -> dict:
        ordered = sorted(group_items, key=lambda c: abs_sec(c["date"], c["start_time"]))
        raw_caps = sorted({
            i
            for it, gi in zip(group_items, global_idxs)
            for i in (it.get("raw_caption_indices") or [gi])
        })
        return {
            "text":                summary.strip(),
            "date":                ordered[0]["date"],
            "start_time":          ordered[0]["start_time"],
            "end_time":            ordered[-1]["end_time"],
            self._SOURCE_KEY:      sorted(set(global_idxs)),
            "raw_caption_indices": raw_caps,
        }

    def segment(self, inputs: List[dict]) -> List[dict]:
        """
        inputs: a time-ordered list of lower-layer nodes. Each must carry
                text, start_time, end_time, date, raw_caption_indices.
        Returns higher-layer nodes sorted by (date, start_time), each carrying
        `self._SOURCE_KEY` = global indices into `inputs`.
        """
        if not inputs:
            return []

        from models.embedding import EmbeddingModel
        em = EmbeddingModel(model_name=_DEDUP_EMBED_MODEL)

        windows = self._build_windows(inputs)
        items_per_window = [[inputs[j] for j in idxs] for idxs in windows]

        # Phase 1: per-window LLM segmentation (overridable for batching).
        results = self._segment_windows(items_per_window)

        # Phase 2: sequential merge — order-dependent (embedding similarity vs
        # already-merged segments), so it stays single-threaded.
        self.window_log = []
        all_segments: List[dict] = []
        seg_vecs: List[np.ndarray] = []

        for wi, (idxs, items, result) in enumerate(tqdm(
                list(zip(windows, items_per_window, results)),
                desc=self.__class__.__name__, unit="win")):
            logger.debug("  [window %d/%d] %d items", wi + 1, len(windows), len(items))

            win_entry: dict = {
                "window_idx": wi,
                "input_range": {
                    "start": f"{items[0]['date']} {items[0]['start_time']}",
                    "end":   f"{items[-1]['date']} {items[-1]['end_time']}",
                },
                "input_count":    len(items),
                "fallback":       result is None,
                "groups":         [],
                "segments_added": [],
            }

            if result is None:
                nodes = [self._make_node(items, idxs,
                                         " ".join(c["text"] for c in items)[:200])]
            else:
                nodes = []
                for group in result.groups:
                    local = [j for j in group.indices if 0 <= j < len(items)]
                    if not local:
                        continue
                    win_entry["groups"].append(
                        {"indices": group.indices, "summary": group.summary})
                    nodes.append(self._make_node(
                        [items[j] for j in local],
                        [idxs[j] for j in local],
                        group.summary,
                    ))

            for node in nodes:
                before = len(all_segments)
                _merge_or_append(
                    self.llm, em, all_segments, seg_vecs, node,
                    index_keys=[self._SOURCE_KEY, "raw_caption_indices"],
                    require_overlap=self._REQUIRE_OVERLAP,
                )
                if len(all_segments) > before:
                    s = all_segments[-1]
                    # Enforce the word count NOW so the log shows the final text.
                    if self._SUMMARY_WORD_MIN and self._SUMMARY_WORD_MAX:
                        ctx = "\n".join(
                            f"({inputs[i]['date']} {inputs[i]['start_time']}-"
                            f"{inputs[i]['end_time']}) {inputs[i]['text']}"
                            for i in s.get("raw_caption_indices", []) if i < len(inputs))
                        new_text = _enforce_word_range(
                            self.llm, s["text"],
                            lo=self._SUMMARY_WORD_MIN, hi=self._SUMMARY_WORD_MAX,
                            context=ctx,
                        )
                        if new_text != s["text"]:
                            s["text"] = new_text
                            mv = em.encode([new_text], batch_size=1)[0]
                            seg_vecs[-1] = mv / max(float(np.linalg.norm(mv)), 1e-9)
                    win_entry["segments_added"].append(s)
                    logger.info("  [%s] %s %s-%s  (%.1fmin) %r",
                                self._LABEL, s["date"], s["start_time"], s["end_time"],
                                _dur_min(s["start_time"], s["end_time"]), s["text"])

            self.window_log.append(win_entry)

        valid = [s for s in all_segments
                 if _time_to_sec(s["start_time"]) <= _time_to_sec(s["end_time"])]
        valid.sort(key=lambda s: (_date_to_day(s["date"]), _time_to_sec(s["start_time"])))

        logger.info("[%s] %d windows → %d nodes",
                    self.__class__.__name__, len(windows), len(valid))
        return valid


class ActivitySegmenter(_CountWindowedSegmenter):
    """
    Raw 30-sec captions → activity nodes  (Human memory, layer 2).

    Splits the raw caption stream into non-overlapping 30-minute windows; within
    each window the LLM partitions the captions into multiple CONTIGUOUS
    activities, each spanning 5–20 minutes, with its own summary. Each activity
    records `raw_caption_indices` back into the caption stream.

    Only the windowing, the prompt, and the merge phase live here — the LLM
    calls themselves are driven by build_activity.py.
    """
    WINDOW_SEC        = 1800   # 30-minute non-overlapping windows over the caption stream
    _SYSTEM           = _SYSTEM_ACTIVITY
    _SOURCE_KEY       = "raw_caption_indices"
    _REQUIRE_OVERLAP  = True
    _ENABLE_THINKING  = False  # non-thinking mode for the activity segmentation call
    _LABEL            = "activity"
    _SUMMARY_WORD_MIN = 10     # enforce activity summary length to 10-20 words
    _SUMMARY_WORD_MAX = 20

    # Hard cap on the per-block duration. Blocks longer than this trigger a re-prompt.
    _MAX_BLOCK_SEC     = 600    # 10 minutes
    _MAX_RETRIES       = 2
