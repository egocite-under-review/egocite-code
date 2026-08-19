"""
EpisodicDAG — igraph multi-level episodic memory graph.

Hierarchy (coarse → fine, edges go parent → child):
  activity     ──► action
  conversation ──► action     (parallel view at the same coarse level)

action nodes are extracted from raw 30-sec captions (multiple per caption).
activity and conversation nodes are derived directly from the captions; both
edge down to every action whose source captions they cover. Each node stores
raw_caption_indices pointing back to the original caption list.
"""

import json
import logging
from typing import Any, Dict, List, Optional

import igraph as ig

logger = logging.getLogger(__name__)

LEVELS = ("activity", "conversation", "action")  # coarse → fine

_ATTRS = ("level", "text", "start_time", "end_time", "date", "raw_caption_indices")


class EpisodicDAG:

    def __init__(self) -> None:
        self.g = ig.Graph(directed=True)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_node(
        self,
        level: str,
        text: str,
        start_time: str,
        end_time: str,
        date: str,
        raw_caption_indices: Optional[List[int]] = None,
    ) -> int:
        """Add a node and return its integer vertex ID."""
        vid = self.g.vcount()
        self.g.add_vertex(
            level=level,
            text=text,
            start_time=start_time,
            end_time=end_time,
            date=date,
            raw_caption_indices=raw_caption_indices or [],
        )
        return vid

    def add_edge(self, parent_id: int, child_id: int) -> None:
        self.g.add_edge(parent_id, child_id)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_node(self, node_id: int) -> Dict[str, Any]:
        v = self.g.vs[node_id]
        return {"id": node_id, **{a: v[a] for a in _ATTRS}}

    def summary(self) -> Dict[str, Any]:
        counts: Dict[str, int] = {lvl: 0 for lvl in LEVELS}
        for v in self.g.vs:
            counts[v["level"]] += 1
        return {"nodes": self.g.vcount(), "edges": self.g.ecount(), **counts}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        data = {
            "vertices": [self.get_node(v.index) for v in self.g.vs],
            "edges": [[e.source, e.target] for e in self.g.es],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
