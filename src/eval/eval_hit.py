"""Retrieval hit rate over a finished set of evaluation runs.

HIT = the FINAL CURATED evidence — the captions the answering agent is actually
grounded on — temporally overlaps the question's ground-truth target time.

    python src/eval/eval_hit.py --benchmark egolifeqa --source densecaption --model qwen
    python src/eval/eval_hit.py --benchmark egomem    --source gemini      --model qwen

Same log set and CLI as ``eval_accuracy.py``: every
``{EGOCITE_OUTPUT}/logs/{benchmark}/{source}/*-{model}.log``. The harness logs
the curated pool once per question,

    [QA] evidence after curation (5): DAY7 13110000-13113000 | DAY7 13050000-...

and the ground truth comes from the dataset, so the rate is recomputed rather
than read off the running counter — an interrupted, resumed, or merged log
still totals correctly.

The overlap rule follows eval/eval_hit_rate.py:

    egolifeqa   target is a POINT   -> hit when start - tol <= target <= end + tol
    egomem      target is a WINDOW  -> hit when the segments overlap it at all

Columns match eval_accuracy.py:

    egolifeqa   Ent.  Evt.  Hab.  Rel.  Task  Avg.
    egomem      Sgl.  Mul.  Det.  Time  Avg.
"""
import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

from eval.eval_accuracy import (ALIASES, ANSI, COLUMNS, HEADER,  # noqa: E402
                                wearer_of)

PERSONS = ["A1_JAKE", "A2_ALICE", "A3_TASHA", "A4_LUCIA", "A5_KATRINA", "A6_SHURE"]
CURATED = re.compile(r"\[QA\] evidence after curation \(\d+\):(.*)")
SEGMENT = re.compile(r"DAY(\d+)\s+(\d+)-(\d+)")


# --- time helpers (from eval/eval_hit_rate.py) ----------------------------
def day_of(s):
    """'DAY6' / 'day6' -> 6."""
    m = re.search(r"(\d+)", str(s))
    return int(m.group(1)) if m else None


def sec_hhmmssff(s):
    """'11152408' -> 11*3600 + 15*60 + 24 (the frame field is dropped)."""
    s = str(s)
    return int(s[0:2]) * 3600 + int(s[2:4]) * 60 + int(s[4:6])


# --- ground truth ---------------------------------------------------------
def load_targets(benchmark):
    """(person, question_id) -> target, as a point for egolifeqa and a
    (start, end) window for egomem."""
    target = {}
    if benchmark == "egolifeqa":
        qa_dir = os.path.join(config.EGOLIFEQA_DATASET, "Ego-QA-4.4K", "manual-2.9K")
        for p in PERSONS:
            f = os.path.join(qa_dir, f"{p}.json")
            if not os.path.exists(f):
                continue
            for q in json.load(open(f)):
                tt = q.get("target_time") or {}
                d, t = day_of(tt.get("date")), tt.get("time")
                if d is not None and t is not None:
                    target[(p, int(q["ID"]))] = (d, sec_hhmmssff(t), None)
        return target

    f = os.path.join(config.EGOMEM_DATASET, "data", "EgoMem-Normalized.json")
    by_name = {e["history_name"].lower(): e for e in json.load(open(f))}
    for p in PERSONS:
        entry = by_name.get(p.split("_")[-1].lower())
        if not entry:
            continue
        # EgoMem questions are unnumbered; the harness numbers them 1..N in order.
        for i, q in enumerate(entry["qa"], start=1):
            ev = q.get("evidence_date") or {}
            d, s, e = day_of(ev.get("date")), ev.get("start_time"), ev.get("end_time")
            if d is not None and s is not None and e is not None:
                target[(p, i)] = (d, sec_hhmmssff(s), sec_hhmmssff(e))
    return target


def is_hit(tgt, segments, tol):
    """segments: [(day, start_sec, end_sec)]. Point rule when the target has no
    end (egolifeqa), interval-overlap rule when it does (egomem)."""
    d, gs, ge = tgt
    for sd, ss, se in segments:
        if sd != d:
            continue
        if ge is None:                       # point target
            if ss - tol <= gs <= se + tol:
                return True
        elif ss <= ge + tol and se >= gs - tol:   # window target
            return True
    return False


def read_log(path):
    """(question_id) -> (category, segments) for one run log. The LAST curated
    line for a question wins, so a resumed run counts the question once."""
    out = {}
    content = open(path, encoding="utf-8", errors="replace").read()
    for block in re.split(r"(?=INFO: \[\d+/\d+\] Q\d+ \()", ANSI.sub("", content)):
        h = HEADER.search(block)
        if not h:
            continue
        cur = CURATED.findall(block)
        if not cur:
            continue                          # question never reached curation
        segs = [(int(m.group(1)), sec_hhmmssff(m.group(2)), sec_hhmmssff(m.group(3)))
                for m in SEGMENT.finditer(cur[-1])]
        cat = ALIASES.get(h.group(2).strip().lower(), h.group(2).strip())
        out[int(h.group(1))] = (cat, segs)
    return out


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--benchmark", required=True, choices=sorted(COLUMNS),
                   help="egolifeqa | egomem")
    p.add_argument("--source", required=True,
                   help="caption source: densecaption | gemini | gemma")
    p.add_argument("--model", required=True,
                   help="model tag in the log filename, e.g. qwen | gpt")
    p.add_argument("--tolerance-sec", type=int, default=0,
                   help="widen the ground-truth window by this many seconds")
    p.add_argument("--per-wearer", action="store_true",
                   help="also print one row per wearer")
    args = p.parse_args()

    log_dir = os.path.join(config.LOG_DIR, args.benchmark, args.source)
    paths = sorted(glob.glob(os.path.join(log_dir, f"*-{args.model}.log")))
    if not paths:
        sys.exit(f"No logs matching *-{args.model}.log in {log_dir}")

    target = load_targets(args.benchmark)
    if not target:
        sys.exit(f"No ground-truth target times found for {args.benchmark}")

    seen = {}
    for path in paths:
        w = wearer_of(path)
        for qid, val in read_log(path).items():
            seen[(w, qid)] = val

    per_cat = defaultdict(lambda: [0, 0])
    per_wearer = defaultdict(lambda: [0, 0])
    no_target = 0
    for (w, qid), (cat, segs) in seen.items():
        tgt = target.get((w, qid))
        if tgt is None:                       # question has no ground-truth time
            no_target += 1
            continue
        hit = is_hit(tgt, segs, args.tolerance_sec)
        per_cat[cat][0] += hit
        per_cat[cat][1] += 1
        per_wearer[w][0] += hit
        per_wearer[w][1] += 1

    cols = COLUMNS[args.benchmark]
    total_hit = sum(v[0] for v in per_wearer.values())
    total_n = sum(v[1] for v in per_wearer.values())

    print(f"{args.benchmark} | source={args.source} | model={args.model}")
    print(f"HIT = curated evidence overlaps the target time "
          f"(tolerance = {args.tolerance_sec}s)")
    print(f"{len(paths)} log(s), {len(per_wearer)} wearer(s), "
          f"{total_n} questions with a ground-truth target"
          + (f" ({no_target} skipped: no target)" if no_target else "") + "\n")

    def pct(c, n):
        return f"{100 * c / n:.2f}" if n else "  -  "

    print("".join(f"{h:>8s}" for h, _ in cols) + f"{'Avg.':>8s}")
    print("".join(f"{pct(*per_cat[c]):>8s}" for _, c in cols)
          + f"{pct(total_hit, total_n):>8s}")
    print("\ncounts: " + "  ".join(
        f"{h} {per_cat[c][0]}/{per_cat[c][1]}" for h, c in cols)
        + f"  |  total {total_hit}/{total_n}")

    if args.per_wearer:
        print()
        for w in sorted(per_wearer):
            c, n = per_wearer[w]
            print(f"  {w:12s} {c:4d}/{n:<4d} = {pct(c, n)}")


if __name__ == "__main__":
    main()
