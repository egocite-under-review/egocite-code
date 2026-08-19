"""
EpisodicSearch — embedding-based search over the episodic DAG.

Features:
  - FAISS per-level semantic search (cosine similarity)
  - Temporal filtering: only retrieve nodes before query_time
  - Question-type routing: start from the most relevant level
  - Graph traversal: drill down from activity → action → raw captions
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

LEVELS = ("action", "activity", "conversation")

# Which level to search first for each question type
QTYPE_PRIMARY_LEVEL = {
    "EntityLog":    "action",     # who/what/where → fine-grained
    "EventRecall":  "activity",   # what happened → mid-level
    "RelationMap":  "activity",   # social interactions → mid-level
    "TaskMaster":   "activity",   # task sequences → mid-level
    "HabitInsight": "activity",   # habits/patterns → mid-level
}


def _sec_of_day(time_str: str) -> int:
    """Seconds-of-day from an HHMMSSFF string."""
    t = str(time_str).zfill(8)
    return int(t[0:2]) * 3600 + int(t[2:4]) * 60 + int(t[4:6])


def _abs_ts(date: str, time_str: str) -> int:
    """Convert DAY{d} + HHMMSSFF to a single sortable integer.

    Case-insensitive on the DAY prefix — some datasets store the query_time
    date lowercase (e.g. "day6"), which must resolve the same as "DAY6".
    """
    day = int(date.upper().replace("DAY", ""))
    return (day - 1) * 86400 + _sec_of_day(time_str)


# ---------------------------------------------------------------------------
# Time-decay for similarity ranking
# ---------------------------------------------------------------------------
# A question may reference a concrete target moment ("DAY2 12PM") or window
# ("DAY2 12PM-2PM"). Embedding cosine scores for nodes inside that window are
# kept as-is; outside it they decay multiplicatively by `lambda` for every hour
# of temporal gap between the node and the window:
#     decayed = cosine * lambda ** (gap_hours)
# `lambda` is configured separately for action and activity levels.

_DEFAULT_DECAY_LAMBDA = 0.99


def _parse_target_window(s: Optional[str]) -> Optional[Tuple[int, int]]:
    """Parse a target time, given in dataset format, into an absolute-second window.

    Dataset format = DAY{n} + a 6/8-digit HHMMSS(FF) time string. Endpoints are
    separated by "to" or "-"; an interval may span days.
      "DAY1 11152408"                       → point             → (t, t)
      "DAY2 12000000 to DAY2 18000000"      → same-day period
      "DAY1 00000000 to DAY3 14300000"      → cross-day period
    Returns (start_abs, end_abs) or None when unparseable.
    """
    if not s:
        return None
    import re as _re
    # Each endpoint is DAYn followed by a 6-8 digit time.
    matches = _re.findall(r"DAY\s*(\d+)\s+(\d{6,8})", s, _re.IGNORECASE)
    if not matches:
        return None
    try:
        # Right-pad to 8 digits (HHMMSS -> HHMMSS00) so the frame field is filled,
        # not the hour (which a left-pad would corrupt).
        pts = [(int(d) - 1) * 86400 + _sec_of_day(t.ljust(8, "0")[:8])
               for d, t in matches]
    except Exception:
        return None
    if not pts:
        return None
    return (min(pts), max(pts))


def target_window(date: str, time: str,
                  end_date: Optional[str] = None,
                  end_time: Optional[str] = None) -> Tuple[int, int]:
    """Build a decay target window from dataset-format timestamps.

    Point : target_window("DAY1", "11152408")
    Period: target_window("DAY1", "11152408", "DAY1", "11302408")
    """
    a = _abs_ts(date, time)
    b = _abs_ts(end_date or date, end_time or time)
    return (min(a, b), max(a, b))


def _interval_gap_seconds(s1: int, e1: int, s2: int, e2: int) -> int:
    """Seconds of gap between two [start, end] intervals; 0 if they overlap."""
    if e1 < s2:
        return s2 - e1
    if e2 < s1:
        return s1 - e2
    return 0


def _decay_factor(node_s: int, node_e: int,
                  target: Tuple[int, int], lam: float) -> float:
    """Multiplicative decay applied to a node's cosine score.

    1.0 inside (or overlapping) the target window; otherwise lam ** gap_hours.
    """
    gap = _interval_gap_seconds(node_s, node_e, target[0], target[1])
    if gap <= 0:
        return 1.0
    return lam ** (gap / 3600.0)


class EpisodicSearch:

    def __init__(
        self,
        dag_file: str,
        raw_captions_file: Optional[str] = None,
        embedding_model_name: str = "Qwen/Qwen3-Embedding-4B",
    ) -> None:
        self.dag_file = dag_file
        self.embedding_model_name = embedding_model_name
        self._em = None
        self._index: Dict[str, Any] = {}
        self._vecs: Dict[str, np.ndarray] = {}   # level → L2-normalised vecs
        self._nodes: Dict[str, List[dict]] = {}
        self._dag_edges: Optional[Dict[int, List[int]]] = None  # parent → [children]
        self._dag_parents: Optional[Dict[int, int]] = None      # child → parent

        logger.info("Loading DAG from %s", dag_file)
        with open(dag_file) as f:
            data = json.load(f)
        all_verts = data.get("vertices", [])
        for level in LEVELS:
            self._nodes[level] = [v for v in all_verts if v.get("level") == level]

        # Build edge lookup in memory (avoid re-loading igraph on every traversal)
        self._dag_edges = {}
        self._dag_parents = {}
        for src, tgt in data.get("edges", []):
            self._dag_edges.setdefault(src, []).append(tgt)
            self._dag_parents[tgt] = src

        # Node id → node dict for fast lookup
        self._id_to_node: Dict[int, dict] = {
            v["id"]: v for v in all_verts
        }

        # Raw captions
        self._raw_captions: List[dict] = []
        if raw_captions_file is None:
            raw_captions_file = os.path.join(
                os.path.dirname(dag_file), "captions", "raw.json"
            )
        if os.path.exists(raw_captions_file):
            with open(raw_captions_file) as f:
                self._raw_captions = json.load(f)

        logger.info(
            "Loaded: %s | raw captions: %d",
            {l: len(self._nodes[l]) for l in LEVELS},
            len(self._raw_captions),
        )

        # Per-day recorded bounds + contiguous-timeline offsets (for time-decay).
        self._build_day_bounds()

    # ------------------------------------------------------------------
    # Contiguous timeline (collapse un-recorded overnight gaps)
    # ------------------------------------------------------------------
    # The dataset records only ~8-12h per day. For time-decay distance we treat
    # consecutive days as contiguous: the last recorded moment of one day sits
    # immediately before the first recorded moment of the next (0-hour overnight
    # gap), while within-day clock gaps are preserved.

    def _build_day_bounds(self) -> None:
        bounds: Dict[int, List[int]] = {}
        for level in LEVELS:
            for n in self._nodes.get(level, []):
                try:
                    day = int(n["date"].upper().replace("DAY", ""))
                    s = _sec_of_day(n["start_time"])
                    e = _sec_of_day(n["end_time"])
                except Exception:
                    continue
                b = bounds.setdefault(day, [s, e])
                if s < b[0]:
                    b[0] = s
                if e > b[1]:
                    b[1] = e
        self._day_start: Dict[int, int] = {d: v[0] for d, v in bounds.items()}
        self._day_dur:   Dict[int, int] = {d: max(0, v[1] - v[0]) for d, v in bounds.items()}
        self._day_offset: Dict[int, int] = {}
        acc = 0
        for d in sorted(bounds):
            self._day_offset[d] = acc
            acc += self._day_dur[d]
        # Absolute recorded intervals (sorted) — used to snap out-of-bound targets.
        self._day_intervals: List[Tuple[int, int]] = sorted(
            ((d - 1) * 86400 + v[0], (d - 1) * 86400 + v[1])
            for d, v in bounds.items()
        )
        if bounds:
            logger.info("Contiguous timeline: %d days, total recorded span=%.2fh",
                        len(bounds), acc / 3600.0)

    def _snap_window(self, window: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """Snap a target window onto recorded time.

        - start → the next recorded timestamp >= start (forward to first available)
        - end   → the previous recorded timestamp <= end (back to last available)
        When recorded time falls within [start, end], returns that snapped window.
        When the window lands entirely in a gap / before / after all recording, it
        collapses to the single CLOSEST recorded timestamp (a degenerate point), so
        decay still applies around the nearest available moment.
        Returns None only when there is no recorded data at all.
        """
        s, e = window
        intervals = self._day_intervals
        if not intervals:
            return None
        # Snap start forward to the next available recorded moment >= s.
        ns: Optional[int] = None
        for a, b in intervals:
            if b < s:                 # interval entirely before s
                continue
            ns = s if a <= s else a   # inside interval → s; else → interval start
            break
        # Snap end backward to the previous available recorded moment <= e.
        ne: Optional[int] = None
        for a, b in reversed(intervals):
            if a > e:                 # interval entirely after e
                continue
            ne = e if e <= b else b   # inside interval → e; else → interval end
            break
        # Recorded time falls within the window → use the snapped span.
        if ns is not None and ne is not None and ns <= ne:
            return (ns, ne)
        # Invalid range: window sits in a gap (or before/after all recording).
        # Collapse to the closest recorded timestamp:
        #   ne = end of the interval just before the window (distance s - ne)
        #   ns = start of the interval just after the window (distance ns - e)
        candidates = []
        if ne is not None:
            candidates.append((s - ne, ne))
        if ns is not None:
            candidates.append((ns - e, ns))
        if not candidates:
            return None
        closest = min(candidates, key=lambda c: c[0])[1]
        return (closest, closest)

    def _to_contiguous(self, abs_ts_val: int) -> int:
        """Map an absolute timestamp to the compressed (overnight-collapsed) timeline.

        Falls back to the raw value when the day has no recorded bounds.
        """
        day = abs_ts_val // 86400 + 1
        sod = abs_ts_val % 86400
        if day not in self._day_offset:
            return abs_ts_val
        within = min(max(sod - self._day_start[day], 0), self._day_dur[day])
        return self._day_offset[day] + within

    # ------------------------------------------------------------------
    # Embedding model (lazy)
    # ------------------------------------------------------------------

    @property
    def em(self):
        if self._em is None:
            from models.embedding import EmbeddingModel
            self._em = EmbeddingModel(model_name=self.embedding_model_name)
        return self._em

    # ------------------------------------------------------------------
    # FAISS index management
    # ------------------------------------------------------------------

    def _index_path(self, level: str) -> Tuple[str, str]:
        base = os.path.dirname(self.dag_file)
        return (
            os.path.join(base, f"faiss_{level}.index"),
            os.path.join(base, f"faiss_{level}_vecs.npy"),
        )

    def _build_index(self, level: str) -> None:
        import faiss
        nodes = self._nodes[level]
        if not nodes:
            return
        idx_path, vecs_path = self._index_path(level)
        if os.path.exists(idx_path) and os.path.exists(vecs_path):
            cached_vecs = np.load(vecs_path)
            if cached_vecs.shape[0] == len(nodes):
                logger.info("Loading cached FAISS index for '%s' (%d nodes)", level, len(nodes))
                self._index[level] = faiss.read_index(idx_path)
                self._vecs[level] = cached_vecs
                return
            logger.warning(
                "Stale cache for '%s' (cached=%d, current=%d nodes) — rebuilding",
                level, cached_vecs.shape[0], len(nodes),
            )

        logger.info("Building FAISS index for '%s' (%d nodes) ...", level, len(nodes))
        # Hard-cap embedding input at 100 words: a single very long node text
        # (e.g. a noscheme action node that fell back to a full multi-sentence
        # caption) forces the whole batch to pad to that length and can OOM the
        # embedder. Truncation affects only the vector, not the stored/displayed
        # text; scheme nodes are already well under 100 words so are unchanged.
        texts = [" ".join(str(n["text"]).split()[:100]) for n in nodes]
        vecs = self.em.encode(texts, batch_size=256).astype(np.float32)
        faiss.normalize_L2(vecs)
        dim = vecs.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(vecs)
        faiss.write_index(index, idx_path)
        np.save(vecs_path, vecs)
        self._index[level] = index
        self._vecs[level] = vecs
        logger.info("  Saved FAISS index → %s", idx_path)

    def _ensure_index(self, level: str) -> None:
        if level not in self._index:
            self._build_index(level)

    def build_all_indexes(self) -> None:
        for level in LEVELS:
            self._ensure_index(level)

    # ------------------------------------------------------------------
    # Temporal filtering
    # ------------------------------------------------------------------

    def _before(self, node: dict, query_abs: int) -> bool:
        """True if node ends before query_abs."""
        return _abs_ts(node["date"], node["end_time"]) <= query_abs

    def _filter_temporal(self, nodes: List[dict], query_abs: int) -> List[dict]:
        return [n for n in nodes if self._before(n, query_abs)]

    # ------------------------------------------------------------------
    # Core search (with temporal gate)
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        level: str = "activity",
        top_k: int = 5,
        before_date: Optional[str] = None,
        before_time: Optional[str] = None,
        target_window: Optional[Tuple[int, int]] = None,
        decay_lambda: Optional[float] = None,
    ) -> List[dict]:
        """
        Semantic search at one level.
        If before_date+before_time are given, only return nodes that end before that point.

        Time-decay (optional): when both `target_window` (start_abs, end_abs) and
        `decay_lambda` are provided, each node's cosine score is multiplied by
        `decay_lambda ** gap_hours`, where gap_hours is the temporal gap (in hours)
        between the node's [start, end] and the target window (0 inside the window,
        i.e. no decay). Ranking and the returned `score` then reflect the decayed
        value; the original cosine is preserved in `base_score`.
        """
        import faiss
        self._ensure_index(level)
        if level not in self._index:
            return []

        nodes = self._nodes[level]
        query_abs = None
        if before_date and before_time:
            query_abs = _abs_ts(before_date, before_time)
            logger.info("search level=%s before=%s %s (abs=%d) total_nodes=%d",
                        level, before_date, before_time, query_abs, len(nodes))
            nodes = self._filter_temporal(nodes, query_abs)

        if not nodes:
            return []

        all_nodes = self._nodes[level]
        full_set = len(nodes) == len(all_nodes)
        if full_set:
            local_idxs = list(range(len(all_nodes)))
        else:
            valid_ids = {n["id"] for n in nodes}
            local_idxs = [i for i, n in enumerate(all_nodes) if n["id"] in valid_ids]

        if not local_idxs:
            return []

        vec = self.em.encode([query], is_query=True).astype(np.float32)
        faiss.normalize_L2(vec)

        # Score only the temporally-valid subset, then take top-k.
        # Avoid a full matrix copy when no temporal filter is active.
        all_vecs = self._vecs[level]
        candidate_vecs = all_vecs if full_set else all_vecs[local_idxs]  # (M, D)
        base_scores = (candidate_vecs @ vec[0]).tolist()  # cosine sim (L2-normed)

        decay_on = target_window is not None and decay_lambda is not None
        if decay_on:
            snapped = self._snap_window(target_window)
            if snapped is None:
                logger.info("search level=%s time-decay disabled "
                            "(target window %s has no recorded time → lambda=1 everywhere)",
                            level, target_window)
                decay_on = False
            else:
                if snapped != target_window:
                    logger.info("search level=%s snapped target window %s → %s",
                                level, target_window, snapped)
                target_window = snapped
                logger.info("search level=%s applying time-decay (lambda=%.3f, window=%s)",
                            level, decay_lambda, target_window)
        scores = []
        for base, idx in zip(base_scores, local_idxs):
            factor = 1.0
            if decay_on:
                factor = self._decay_for(all_nodes[idx], target_window, decay_lambda)
            scores.append(base * factor)

        ranked = sorted(zip(scores, base_scores, local_idxs),
                        key=lambda t: t[0], reverse=True)
        results = []
        for score, base, idx in ranked[:top_k]:
            node = dict(all_nodes[idx])
            node["score"] = float(score)
            node["base_score"] = float(base)
            results.append(node)

        logger.info("search level=%s pool=%d top_k=%d returned=%d", level, len(local_idxs), top_k, len(results))
        return results

    def _decay_for(self, node: dict, target_window: Tuple[int, int], lam: float) -> float:
        """Time-decay multiplier for a node given a target window. 1.0 on parse error.

        The gap is measured on the contiguous (overnight-collapsed) timeline so the
        decay reflects recorded-time distance, not wall-clock distance across days.
        """
        try:
            ns = self._to_contiguous(_abs_ts(node["date"], node["start_time"]))
            ne = self._to_contiguous(_abs_ts(node["date"], node["end_time"]))
        except Exception:
            return 1.0
        tw = (self._to_contiguous(target_window[0]),
              self._to_contiguous(target_window[1]))
        return _decay_factor(ns, ne, tw, lam)

    def search_within(
        self,
        query: str,
        node_ids: List[int],
        level: str = "action",
        top_k: int = 30,
        target_window: Optional[Tuple[int, int]] = None,
        decay_lambda: Optional[float] = None,
    ) -> List[dict]:
        """
        Score a specific subset of nodes (at `level`) against the query using stored
        vectors, return top_k by cosine similarity with optional time decay.
        """
        import faiss
        self._ensure_index(level)
        if level not in self._vecs:
            return []

        all_nodes = self._nodes[level]
        id_to_idx = {n["id"]: i for i, n in enumerate(all_nodes)}
        local_idxs = [id_to_idx[nid] for nid in node_ids if nid in id_to_idx]
        if not local_idxs:
            return []

        vec = self.em.encode([query], is_query=True).astype(np.float32)
        faiss.normalize_L2(vec)

        candidate_vecs = self._vecs[level][local_idxs]
        base_scores = (candidate_vecs @ vec[0]).tolist()

        decay_on = target_window is not None and decay_lambda is not None
        if decay_on:
            snapped = self._snap_window(target_window)
            if snapped is None:
                logger.info("search level=%s time-decay disabled "
                            "(target window %s has no recorded time → lambda=1 everywhere)",
                            level, target_window)
                decay_on = False
            else:
                if snapped != target_window:
                    logger.info("search level=%s snapped target window %s → %s",
                                level, target_window, snapped)
                target_window = snapped
                logger.info("search level=%s applying time-decay (lambda=%.3f, window=%s)",
                            level, decay_lambda, target_window)
        scored = []
        for base, idx in zip(base_scores, local_idxs):
            factor = self._decay_for(all_nodes[idx], target_window, decay_lambda) if decay_on else 1.0
            scored.append((base * factor, base, idx))

        results = []
        for score, base, idx in sorted(scored, key=lambda t: t[0], reverse=True)[:top_k]:
            node = dict(all_nodes[idx])
            node["score"] = float(score)
            node["base_score"] = float(base)
            results.append(node)

        logger.info("search level=%s pool=%d top_k=%d returned=%d", level, len(local_idxs), top_k, len(results))
        return results

    # ------------------------------------------------------------------
    # Graph traversal
    # ------------------------------------------------------------------

    def get_children(self, node_id: int) -> List[dict]:
        return [
            self._id_to_node[c]
            for c in self._dag_edges.get(node_id, [])
            if c in self._id_to_node
        ]

    def get_parent(self, node_id: int) -> Optional[dict]:
        pid = self._dag_parents.get(node_id)
        return self._id_to_node.get(pid) if pid is not None else None

    def get_node(self, node_id: int) -> Optional[dict]:
        return self._id_to_node.get(node_id)

    def drill_to_actions(self, node_id: int) -> List[dict]:
        """Recursively collect all action-level nodes under node_id."""
        node = self._id_to_node.get(node_id)
        if node is None:
            return []
        if node["level"] == "action":
            return [node]
        result = []
        for child_id in self._dag_edges.get(node_id, []):
            result.extend(self.drill_to_actions(child_id))
        return result

    def get_raw_captions(self, node: dict) -> List[dict]:
        indices = node.get("raw_caption_indices") or []
        return [self._raw_captions[i] for i in indices if i < len(self._raw_captions)]

    def find_at_time(self, date: str, time_str: str) -> Dict[str, List[dict]]:
        """
        Return every activity node, action node and raw caption whose
        [start_time, end_time] (on the given date) contains the given timestamp.
        """
        try:
            target_abs = _abs_ts(date, time_str)
        except Exception:
            return {"activities": [], "actions": [], "captions": []}

        def _covers(item: dict) -> bool:
            if item.get("date") != date:
                return False
            try:
                s = _abs_ts(item["date"], item["start_time"])
                e = _abs_ts(item["date"], item["end_time"])
            except Exception:
                return False
            return s <= target_abs <= e

        activities = [n for n in self._nodes.get("activity", []) if _covers(n)]
        actions    = [n for n in self._nodes.get("action", []) if _covers(n)]
        captions   = [c for c in self._raw_captions if _covers(c)]
        return {"activities": activities, "actions": actions, "captions": captions}

    # ------------------------------------------------------------------
    # High-level retrieval: question-type aware, with drill-down
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        qtype: str = "EventRecall",
        before_date: Optional[str] = None,
        before_time: Optional[str] = None,
        top_k: int = 5,
    ) -> Dict[str, List[dict]]:
        """
        Full retrieval pipeline:
          1. Route to primary level based on question type
          2. Search primary level (temporally gated)
          3. For each hit, drill down to action nodes
          4. Search remaining levels for broader context
          5. Return structured results by level
        """
        primary = QTYPE_PRIMARY_LEVEL.get(qtype, "activity")
        other_levels = [l for l in LEVELS if l != primary]

        results: Dict[str, List[dict]] = {}

        # Primary level search
        primary_hits = self.search(
            query, level=primary, top_k=top_k,
            before_date=before_date, before_time=before_time,
        )
        results[primary] = primary_hits

        # Drill down from primary hits to action level (only if primary != action)
        if primary != "action" and primary_hits:
            action_nodes = []
            seen = set()
            for hit in primary_hits:
                for action in self.drill_to_actions(hit["id"]):
                    if action["id"] not in seen:
                        seen.add(action["id"])
                        action_nodes.append(action)
            if action_nodes:
                results["action_drilled"] = action_nodes[:top_k * 3]

        # Secondary search on remaining levels (fewer results)
        for level in other_levels:
            hits = self.search(
                query, level=level, top_k=max(2, top_k // 2),
                before_date=before_date, before_time=before_time,
            )
            if hits:
                results[level] = hits

        return results
