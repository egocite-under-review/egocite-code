"""Aggregate accuracy over a finished set of evaluation runs.

Reads every run log written by the eval harnesses for one (benchmark, source,
model) combination and reports accuracy per question category, per wearer, and
overall.

    python src/eval/eval_accuracy.py --benchmark egolifeqa --source densecaption --model qwen
    python src/eval/eval_accuracy.py --benchmark egomem    --source gemini      --model qwen
    python src/eval/eval_accuracy.py --benchmark egor1     --source densecaption --model gpt --split manual

Logs are read from ``{EGOCITE_OUTPUT}/logs/{benchmark}/{source}/*-{model}.log``,
the same place the harnesses write them. Two per-question lines carry
everything needed:

    [12/475] Q37 (EntityLog): Who used the screwdriver first?
    ✓  pred=B  gt=B  [8/12 = 66.7%]  ...

so accuracy is recomputed from the runs themselves rather than trusting a
summary line — a log that was interrupted, resumed, or merged still totals
correctly. Questions are keyed by (wearer, question id) and the LAST verdict
wins, so a resumed run that repeats a question is counted once.

Column sets are fixed per benchmark:

    egolifeqa   Ent.  Evt.  Hab.  Rel.  Task  Avg.
    egor1       Ent.  Evt.  Hab.  Rel.  Task  Avg.   (--split manual|gemini)
    egomem      Sgl.  Mul.  Det.  Time  Avg.
"""
import argparse
import glob
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

# --- benchmark column definitions -----------------------------------------
# (header, canonical category). Column order is the report's column order.
COLUMNS = {
    "egolifeqa": [("Ent.",  "EntityLog"),
                  ("Evt.",  "EventRecall"),
                  ("Hab.",  "HabitInsight"),
                  ("Rel.",  "RelationMap"),
                  ("Task",  "TaskMaster")],
    "egor1":     [("Ent.",  "EntityLog"),
                  ("Evt.",  "EventRecall"),
                  ("Hab.",  "HabitInsight"),
                  ("Rel.",  "RelationMap"),
                  ("Task",  "TaskMaster")],
    "egomem":    [("Sgl.",  "single_event"),
                  ("Mul.",  "multi_event"),
                  ("Det.",  "event_detail"),
                  ("Time",  "time_query")],
}

# The released question sets label the same category several ways — English
# variants, singular/plural, and untranslated Chinese. Map them all onto the
# canonical name so a wearer's questions are not split across columns.
ALIASES = {
    # egolifeqa
    "entity log": "EntityLog", "entitylog": "EntityLog", "实体日志": "EntityLog",
    "event memory": "EventRecall", "event recall": "EventRecall",
    "event recollection": "EventRecall", "eventrecall": "EventRecall",
    "事件回忆": "EventRecall",
    "behavior habit": "HabitInsight", "behavior habits": "HabitInsight",
    "behavioral habits": "HabitInsight", "habitinsight": "HabitInsight",
    "行为习惯": "HabitInsight",
    "interpersonal relationships": "RelationMap", "relationmap": "RelationMap",
    "人际关系": "RelationMap",
    "future plan": "TaskMaster", "future plans": "TaskMaster",
    "taskmaster": "TaskMaster", "未来计划": "TaskMaster",
    # egomem
    "single_event": "single_event", "multi_event": "multi_event",
    "event_detail": "event_detail", "time_query": "time_query",
}

ANSI = re.compile(r"\x1b\[[0-9;]*m")
HEADER = re.compile(r"INFO: \[\d+/\d+\] Q(\d+) \(([^)]+)\):")
VERDICT = re.compile(r"INFO: ([✓✗])  pred=")
# The wearer id inside {timestamp}-{benchmark}-{PERSON}-{model}.log. Matched by
# shape rather than by field position: some timestamps use a dash (20260808-
# 041323) instead of an underscore, which shifts every field along.
FNAME = re.compile(r"(A\d+_[A-Za-z]+)")


def wearer_of(path):
    """Recover the wearer from a harness log filename, else use the filename."""
    m = FNAME.search(os.path.basename(path))
    return m.group(1) if m else os.path.basename(path)


def read_log(path):
    """Yield (question_id, canonical_category) -> correct for one run log."""
    out = {}
    qid = cat = None
    for line in open(path, errors="replace"):
        line = ANSI.sub("", line)
        m = HEADER.search(line)
        if m:
            qid = int(m.group(1))
            cat = ALIASES.get(m.group(2).strip().lower(), m.group(2).strip())
            continue
        if qid is None:
            continue
        m = VERDICT.search(line)
        if m:
            # Last verdict for a question wins — resumed runs repeat questions.
            out[qid] = (cat, m.group(1) == "✓")
            qid = cat = None
    return out


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--benchmark", required=True, choices=sorted(COLUMNS),
                   help="egolifeqa | egor1 | egomem")
    p.add_argument("--source", required=True,
                   help="caption source: densecaption | gemini | gemma")
    p.add_argument("--model", required=True,
                   help="model tag in the log filename, e.g. qwen | gpt")
    p.add_argument("--split", choices=["manual", "gemini"], default=None,
                   help="egor1 only: restrict to one benchmark half "
                        "(default: both, pooled)")
    p.add_argument("--per-wearer", action="store_true",
                   help="also print one row per wearer")
    args = p.parse_args()

    log_dir = os.path.join(config.LOG_DIR, args.benchmark, args.source)
    pattern = (f"*-{args.split}-*-{args.model}.log" if args.split
               else f"*-{args.model}.log")
    paths = sorted(glob.glob(os.path.join(log_dir, pattern)))
    if not paths:
        sys.exit(f"No logs matching *-{args.model}.log in {log_dir}")

    # (wearer, qid) -> (category, correct); later logs override earlier ones so a
    # merged/complete log supersedes the partial runs it was built from.
    seen = {}
    for path in paths:
        w = wearer_of(path)
        # The two Ego-R1 halves reuse question ids, so the split is part of the key.
        half = next((x for x in ("manual", "gemini") if f"-{x}-" in os.path.basename(path)), "")
        for qid, val in read_log(path).items():
            seen[(w, half, qid)] = val

    per_cat = defaultdict(lambda: [0, 0])           # canonical -> [correct, n]
    per_wearer = defaultdict(lambda: [0, 0])
    unknown = defaultdict(int)
    for (w, _, _), (cat, ok) in seen.items():
        per_cat[cat][0] += ok
        per_cat[cat][1] += 1
        per_wearer[w][0] += ok
        per_wearer[w][1] += 1
    known = {c for _, c in COLUMNS[args.benchmark]}
    for cat, (_, n) in per_cat.items():
        if cat not in known:
            unknown[cat] += n

    cols = COLUMNS[args.benchmark]
    total_ok = sum(v[0] for v in per_wearer.values())
    total_n = sum(v[1] for v in per_wearer.values())

    print(f"{args.benchmark} | source={args.source} | model={args.model}"
          + (f" | split={args.split}" if args.split else ""))
    print(f"{len(paths)} log(s), {len(per_wearer)} wearer(s), {total_n} questions\n")

    def pct(c, n):
        return f"{100 * c / n:.2f}" if n else "  -  "

    head = "".join(f"{h:>8s}" for h, _ in cols) + f"{'Avg.':>8s}"
    print(head)
    row = "".join(f"{pct(*per_cat[c]):>8s}" for _, c in cols)
    print(row + f"{pct(total_ok, total_n):>8s}")

    print("\ncounts: " + "  ".join(
        f"{h} {per_cat[c][0]}/{per_cat[c][1]}" for h, c in cols)
        + f"  |  total {total_ok}/{total_n}")

    if args.per_wearer:
        print()
        for w in sorted(per_wearer):
            c, n = per_wearer[w]
            print(f"  {w:12s} {c:4d}/{n:<4d} = {pct(c, n)}")

    if unknown:
        print("\nWARNING: categories outside the "
              f"{args.benchmark} column set (counted in Avg. only):")
        for cat, n in sorted(unknown.items(), key=lambda kv: -kv[1]):
            print(f"  {cat!r}: {n}")


if __name__ == "__main__":
    main()
