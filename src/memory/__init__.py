"""Episodic memory: construction and representation.

Build pipeline (each stage is also runnable as a script — see src/scripts/build_memory.sh):

    build_action        raw 30-sec captions -> atomic action nodes
    build_activity      captions            -> activity nodes
                        (segment -> retry -> disentangle -> merge)
    build_conversation  captions            -> conversation-topic nodes
    build               action + activity + conversation -> dag.json

All prompts used by these stages live in prompt/memory/*.txt.
Everything is written under output/memory/{source}/{person}/ (see config.py).
Reading that memory back is the agent's job — see agent/search.py.
"""

from memory.dag import EpisodicDAG, LEVELS
from memory.segmentation import ActionExtractor, ActivitySegmenter

__all__ = ["EpisodicDAG", "LEVELS", "ActionExtractor", "ActivitySegmenter"]
