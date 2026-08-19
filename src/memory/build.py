"""
Build the episodic memory DAG for a given person.

Pipeline
--------
0. build_action.py    extracts atomic actions per caption     → action.json
1. build_activity.py  groups captions into 5-min activities   → activity.json
2. build.py           DAG assembly (this script)              → dag.json

This script consumes action.json and activity.json from disk.

Run from long-memory/ root:
    python episodic/build.py --person A1_JAKE
"""

import argparse
import datetime
import json
import logging
import os
import sys
import time

import tqdm as _tqdm_module

_orig_tqdm_init = _tqdm_module.tqdm.__init__
def _silent_tqdm_init(self, *args, **kwargs):
    kwargs["disable"] = True
    _orig_tqdm_init(self, *args, **kwargs)
_tqdm_module.tqdm.__init__ = _silent_tqdm_init

from tqdm import tqdm as _tqdm_base
class tqdm(_tqdm_base):
    def __init__(self, *args, **kwargs):
        kwargs.pop("disable", None)
        _orig_tqdm_init(self, *args, **kwargs)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("accelerate").setLevel(logging.WARNING)
logger = logging.getLogger("episodic.build")

# src/memory/<this file>  ->  src/  (so the flat `config` / `prompts` / `memory`
# modules resolve when this file is run directly as a script).
_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import config  # noqa: E402

from memory.dag import EpisodicDAG


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build episodic memory DAG.")
    p.add_argument("--person", required=True, help="Person ID, e.g. A1_JAKE")
    p.add_argument("--source", default="densecaption", choices=config.SOURCES,
                   help="Caption source: 'densecaption' uses the released EgoLife captions, "
                        "'gemini'/'gemma' use the VLM-generated ones.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    person = args.person

    out_dir       = config.memory_dir(person, args.source, create=True)
    caps_dir      = config.captions_dir(person, args.source)
    action_file       = os.path.join(caps_dir, "action.json")
    activity_file     = os.path.join(caps_dir, "activity.json")
    conversation_file = os.path.join(caps_dir, "conversation.json")
    dag_file          = os.path.join(out_dir, "dag.json")

    for f, hint in ((action_file, "build_action.py"),
                    (activity_file, "build_activity.py"),
                    (conversation_file, "build_conversation.py")):
        if not os.path.exists(f):
            logger.error("Not found: %s  (run %s first)", f, hint)
            sys.exit(1)

    with open(action_file) as f:
        action_nodes = json.load(f)
    logger.info("Loaded %d action nodes", len(action_nodes))

    with open(activity_file) as f:
        activity_nodes = json.load(f)
    logger.info("Loaded %d activity nodes", len(activity_nodes))

    with open(conversation_file) as f:
        conversation_nodes = json.load(f)
    logger.info("Loaded %d conversation nodes", len(conversation_nodes))

    # ------------------------------------------------------------------
    # Build DAG  activity ──► action
    # ------------------------------------------------------------------
    logger.info("=== Building DAG ===")
    dag = EpisodicDAG()
    t0 = time.time()

    cap_to_action_vids: dict = {}
    for node in tqdm(action_nodes, desc="Action nodes", unit="node"):
        vid = dag.add_node(
            level="action",
            text=node["text"],
            start_time=node["start_time"],
            end_time=node["end_time"],
            date=node["date"],
            raw_caption_indices=node.get("raw_caption_indices", []),
        )
        for cap_idx in node.get("raw_caption_indices", []):
            cap_to_action_vids.setdefault(cap_idx, []).append(vid)

    for node in tqdm(activity_nodes, desc="Activity nodes", unit="node"):
        raw_idx = sorted(set(node.get("raw_caption_indices", [])))
        vid = dag.add_node(
            level="activity",
            text=node["text"],
            start_time=node["start_time"],
            end_time=node["end_time"],
            date=node["date"],
            raw_caption_indices=raw_idx,
        )
        for cap_idx in raw_idx:
            for action_vid in cap_to_action_vids.get(cap_idx, []):
                dag.add_edge(vid, action_vid)

    for node in tqdm(conversation_nodes, desc="Conversation nodes", unit="node"):
        dag.add_node(
            level="conversation",
            text=node["text"],
            start_time=node["start_time"],
            end_time=node["end_time"],
            date=node["date"],
            raw_caption_indices=node.get("raw_caption_indices", []),
        )

    elapsed = time.time() - t0
    dag.save(dag_file)
    s = dag.summary()
    logger.info(
        "DAG saved: %d nodes (%d activity, %d action, %d conversation), %d edges → %s  (%.1fs)",
        s["nodes"], s["activity"], s["action"], s.get("conversation", 0), s["edges"], dag_file, elapsed,
    )

    build_log = {
        "person": person,
        "build_time": datetime.datetime.now().isoformat(),
        "action_count": len(action_nodes),
        "activity_count": len(activity_nodes),
        "conversation_count": len(conversation_nodes),
        "dag": {**s, "file": dag_file, "seconds": elapsed},
    }
    log_file = os.path.join(out_dir, "build_log.json")
    with open(log_file, "w") as f:
        json.dump(build_log, f, indent=2, ensure_ascii=False)
    logger.info("Build log → %s", log_file)
    logger.info("=== Done → %s ===", out_dir)


if __name__ == "__main__":
    main()
