"""
Evaluate a range of questions from the EgoLife QA benchmark using the episodic DAG.

Retrieval: ReAct search loop — LLM issues search queries, harness executes them.
Answering: Qwen3.6-27B reads collected evidence and picks A/B/C/D.

Run from long-memory/ root:
    python script/3-eval-qa.py --person A1_JAKE --n-start 0 --n-end 20
"""

import argparse
import datetime
import json
import logging
import os
import re
import sys
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("accelerate").setLevel(logging.WARNING)
logger = logging.getLogger("eval")

# src/eval/<this file>  ->  src/  (so the flat `config` / `prompts` / `agent`
# modules resolve when this file is run directly as a script).
_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import config  # noqa: E402
import prompts  # noqa: E402

QA_FILE = os.path.join(config.EGOR1_DATASET,
                       "{benchmark}-benchmark", "{person}.json")

SYSTEM_PROMPT = prompts.load("eval/qa_system")


def build_context(result: dict, query_date: str, query_time: str) -> str:
    lines = [f"[Memory query point: {query_date} {query_time}]\n"]

    if result.get("raw_captions"):
        lines.append("\n--- RAW OBSERVATIONS (30-sec captions, chronological) ---")
        for cap in result["raw_captions"]:
            text = cap.get("text") or cap.get("caption", "")
            ts = f"{cap.get('date', '')} {cap.get('start_time', '')}–{cap.get('end_time', '')}"
            lines.append(f"  [{ts}] {text}")

    return "\n".join(lines)


def build_prompt(q: dict, context: str) -> str:
    return (
        f"{context}\n\n"
        f"Question: {q['question']}\n"
        f"A) {q['choice_a']}\n"
        f"B) {q['choice_b']}\n"
        f"C) {q['choice_c']}\n"
        f"D) {q['choice_d']}\n\n"
        f"Answer (A/B/C/D only):"
    )


def extract_answer(text: str) -> str:
    # Strip thinking block so reasoning content doesn't interfere
    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Explicit answer patterns (highest confidence)
    for pattern in [
        r"(?:answer(?:\s+is)?|option|choice)[:\s]+([A-D])\b",  # "answer is C", "option C"
        r"\b([A-D])\s*[\)\.]",                                  # "C)" or "C."
        r"^\s*([A-D])\s*$",                                     # bare letter on its own line
    ]:
        m = re.search(pattern, clean, re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).upper()

    # Fallback: last A/B/C/D in the cleaned text (answer typically comes after reasoning)
    letters = re.findall(r"\b([A-D])\b", clean.upper())
    if letters:
        return letters[-1]

    return "?"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--person", default="A1_JAKE")
    p.add_argument("--n-start", type=int, default=0, help="Start index (inclusive)")
    p.add_argument("--n-end",   type=int, default=20, help="End index (exclusive)")
    p.add_argument("--all", action="store_true",
                   help="Evaluate ALL questions for the person (ignores --n-start/--n-end).")
    p.add_argument("--qids", type=int, nargs="+", default=None,
                   help="Explicit list of question IDs to evaluate (space-separated). "
                        "When given, overrides --n-start/--n-end; questions are run in "
                        "the order listed and unknown IDs are skipped with a warning.")
    p.add_argument("--llm-name", default="sonnet",
                   choices=["sonnet", "opus", "qwen", "gpt"],
                   help="Model config to use: 'sonnet' → MODEL_CONFIG_SONNET_4_6 "
                        "(claude-sonnet-4-6 via Anthropic API), "
                        "'opus' → MODEL_CONFIG_OPUS_4_8 "
                        "(claude-opus-4-8 via Anthropic API), "
                        "'qwen' → MODEL_CONFIG_QWEN_3_6_27B "
                        "(Qwen/Qwen3.6-27B-FP8 via local vLLM at localhost:8001), "
                        "'gpt' → MODEL_CONFIG_GPT_5_4 (gpt-5.4 via the OpenAI "
                        "Responses API; needs OPENAI_API_KEY).")
    p.add_argument("--max-rounds", type=int, default=5,
                   help="Max search rounds per question.")
    p.add_argument("--top-k", type=int, default=5,
                   help="Results per search call.")
    p.add_argument("--lambda-action", type=float, default=0.99,
                   help="Per-hour time-decay base for ACTION similarity scores "
                        "(applied only when the agent emits a resolvable time_query).")
    p.add_argument("--lambda-activity", type=float, default=0.99,
                   help="Per-hour time-decay base for ACTIVITY similarity scores.")
    p.add_argument("--no-time-decay", action="store_true",
                   help="Disable time decay entirely (ranking uses raw cosine; the "
                        "agent still extracts time_query for the caption reranker).")
    p.add_argument("--out-name", default=None,
                   help="Basename (no extension) for the output .log/.json files. "
                        "Default: {YYYYMMDD_HHMMSS}-{llm-name}. Files are written to "
                        "data/episodic/{person}/ or data/vlm_episodic/{person}/ depending on --source.")
    p.add_argument("--raw-captions-file", default=None,
                   help="Override the 30-sec caption list used to expand evidence. "
                        "Default: captions/raw.json beside the memory's dag.json.")
    p.add_argument("--source", default="densecaption", choices=config.SOURCES,
                   help="Memory source: 'groundtruth' uses data/episodic/ (default), "
                        "'gemini'/'gemma' use the matching VLM memory.")
    p.add_argument("--benchmark", default="manual", choices=["manual", "gemini"],
                   help="Ego-R1-Bench split: 'manual' (manual-benchmark) or "
                        "'gemini' (gemini-benchmark). Default: manual.")
    p.add_argument("--vllm-endpoint", default=config.VLLM_BASE_URL,
                   help="Base URL of the vLLM server for Qwen models (--llm-name qwen). "
                        "A bare host:port is accepted and normalized (http:// and /v1 "
                        "added). Ignored for sonnet/opus. Default: http://localhost:8001/v1.")
    args = p.parse_args()

    # Normalize the vLLM endpoint: accept "localhost:8001", "http://host:8001",
    # or a full ".../v1" URL and coerce to the scheme+/v1 form VLLMClient expects.
    vllm_endpoint = args.vllm_endpoint.strip().rstrip("/")
    if not vllm_endpoint.startswith(("http://", "https://")):
        vllm_endpoint = "http://" + vllm_endpoint
    if not vllm_endpoint.endswith("/v1"):
        vllm_endpoint = vllm_endpoint + "/v1"
    args.vllm_endpoint = vllm_endpoint

    qa_file  = QA_FILE.format(benchmark=args.benchmark, person=args.person)
    dag_file = config.dag_file(args.person, args.source)
    # None -> EpisodicSearch reads captions/raw.json beside the dag.
    raw_captions_file = args.raw_captions_file
    if raw_captions_file:
        logger.info("Raw QA captions: %s", raw_captions_file)

    # Default basename = {timestamp}-{bench}-{person}-{model} so a results
    # directory can be aggregated across persons and reruns never collide.
    # Override with --out-name to set a custom tag.
    if args.out_name:
        out_basename = args.out_name
    else:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # Sanitize the model name for use as a filename component.
        model_slug = args.llm_name  # "sonnet" or "qwen"
        out_basename = f"{ts}-egor1-{args.benchmark}-{args.person}-{model_slug}"

    # Mirror all log output to a file alongside the eval JSON
    log_file = config.log_file(f"{out_basename}.log", os.path.join("egor1", args.source))
    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S"))
    logging.getLogger().addHandler(file_handler)
    logger.info("Logging to %s", log_file)

    with open(qa_file) as f:
        all_questions = json.load(f)
    if args.qids:
        by_id = {q["ID"]: q for q in all_questions}
        questions = [by_id[qid] for qid in args.qids if qid in by_id]
        missing = [qid for qid in args.qids if qid not in by_id]
        if missing:
            logger.warning("Question IDs not found and skipped: %s", missing)
        logger.info("Loaded %d questions by explicit IDs: %s",
                    len(questions), [q["ID"] for q in questions])
    elif args.all:
        questions = all_questions
        logger.info("Loaded ALL %d questions", len(questions))
    else:
        questions = all_questions[args.n_start:args.n_end]
        logger.info("Loaded %d questions (index %d:%d)",
                    len(questions), args.n_start, args.n_end)

    import agent.agent as _agent_mod
    from agent.agent import EpisodicAgent

    # Select and activate the MODEL_CONFIG based on --llm-name.
    if args.llm_name == "sonnet":
        _agent_mod.MODEL_CONFIG = _agent_mod.MODEL_CONFIG_SONNET_4_6
    elif args.llm_name == "opus":
        _agent_mod.MODEL_CONFIG = _agent_mod.MODEL_CONFIG_OPUS_4_8
    elif args.llm_name == "gpt":
        _agent_mod.MODEL_CONFIG = _agent_mod.MODEL_CONFIG_GPT_5_4
    else:
        _agent_mod.MODEL_CONFIG = _agent_mod.MODEL_CONFIG_QWEN_3_6_27B
    cfg = _agent_mod.MODEL_CONFIG
    logger.info("Active MODEL_CONFIG: %s", args.llm_name)
    if args.llm_name == "qwen":
        logger.info("  vLLM endpoint   : %s", args.vllm_endpoint)
    logger.info("  retrieval_agent : model=%s  max_tokens=%s  enable_thinking=%s  tool_call=%s",
                cfg["retrieval_agent"]["model"], cfg["retrieval_agent"].get("max_tokens"),
                cfg["retrieval_agent"].get("enable_thinking"), cfg["retrieval_agent"]["tool_call"])
    logger.info("  curation_agent  : model=%s  max_tokens=%s  enable_thinking=%s  tool_call=%s",
                cfg["curation_agent"]["model"], cfg["curation_agent"].get("max_tokens"),
                cfg["curation_agent"].get("enable_thinking"), cfg["curation_agent"]["tool_call"])
    logger.info("  answering_agent : model=%s  max_tokens=%s  enable_thinking=%s  tool_call=%s",
                cfg["answering_agent"]["model"], cfg["answering_agent"].get("max_tokens"),
                cfg["answering_agent"].get("enable_thinking"), cfg["answering_agent"]["tool_call"])

    # Build LLM instances from the active config.
    # All per-call settings (enable_thinking, effort) come from MODEL_CONFIG at
    # call time — only model_name and max_tokens are baked into the instance.
    def _make_llm_from_cfg(role: str):
        role_cfg  = cfg[role]
        model     = role_cfg["model"]
        max_tok   = role_cfg.get("max_tokens", 4096)
        n = model.lower()
        if any(tag in n for tag in ("claude", "sonnet", "opus", "haiku")):
            from models.claude_api import ClaudeAPI
            effort = role_cfg.get("effort", "medium")
            return ClaudeAPI(model_name=model, max_tokens=max_tok, effort=effort), f"ClaudeAPI({model})"
        if "gpt" in n:
            # OpenAI Responses API. Per-call reasoning effort still comes from
            # MODEL_CONFIG (the agent passes it at call time); only model_name and
            # max_tokens are baked into the instance. Needs OPENAI_API_KEY.
            from models.openai_gpt import GPTAPI
            return (GPTAPI(model_name=model, max_tokens=max_tok,
                           reasoning_effort=role_cfg.get("effort")),
                    f"GPTAPI({model})")
        from models.vllm_client import VLLMClient
        return (VLLMClient(model_name=model, max_tokens=max_tok, base_url=args.vllm_endpoint),
                f"VLLMClient({model} @ {args.vllm_endpoint})")

    llm,          _ = _make_llm_from_cfg("retrieval_agent")
    retriever_llm, _ = _make_llm_from_cfg("helper_agent")

    agent = EpisodicAgent(llm=llm, retriever_llm=retriever_llm,
                          dag_file=dag_file,
                          raw_captions_file=raw_captions_file,
                          max_rounds=args.max_rounds, top_k=args.top_k,
                          lambda_action=args.lambda_action,
                          lambda_activity=args.lambda_activity,
                          enable_time_decay=not args.no_time_decay)
    if args.no_time_decay:
        logger.info("Time-decay: DISABLED (--no-time-decay)")
    else:
        logger.info("Time-decay lambdas: action=%.3f activity=%.3f",
                    args.lambda_action, args.lambda_activity)

    results = []
    correct = 0
    # Caption-hit accounting (only counted for questions with a valid GT target_time).
    NEAR_GT_SEC              = 5 * 60   # ±5 minutes — same as the agent's "Next to GT" band
    gt_q_count               = 0        # questions with a valid target_time
    pre_hit_count            = 0        # HIT  before curation (deduped pool)
    pre_near_count           = 0        # NEAR before curation (deduped pool)
    post_hit_count           = 0        # HIT  after  curation (curated pool)
    post_near_count          = 0        # NEAR after  curation (curated pool)
    from agent.search import _abs_ts

    for i, q in enumerate(questions):
        qid        = q["ID"]
        qtype      = q["type"]
        answer     = q["answer"]
        query_date = q["query_time"]["date"]
        query_time = q["query_time"]["time"]

        target_date = q.get("target_time", {}).get("date", "?")
        target_time = q.get("target_time", {}).get("time", "?")

        logger.info("\n" + "="*60)
        logger.info("[%d/%d] Q%d (%s): %s", i+1, len(questions), qid, qtype, q["question"])
        logger.info("  query_time=%s %s  *** target_time=%s %s ***",
                    query_date, query_time, target_date, target_time)

        if target_date != "?" and target_time != "?":
            gt = agent._search.find_at_time(target_date, target_time)
            for n in gt.get("activities", []):
                logger.info("  *** GT activity *** [%s %s-%s] %s",
                            n["date"], n["start_time"], n["end_time"],
                            n.get("text") or n.get("caption", ""))
            for a in gt["actions"]:
                logger.info("  *** GT action  *** [%s %s-%s] %s",
                            a["date"], a["start_time"], a["end_time"], a["text"])
            for c in gt["captions"]:
                logger.info("  *** GT caption *** [%s %s-%s] %s",
                            c["date"], c["start_time"], c["end_time"],
                            c.get("text") or c.get("caption", ""))
            if not gt.get("activities") and not gt["actions"] and not gt["captions"]:
                logger.info("  *** GT *** (no activity/action/caption covers target_time)")

        # ReAct search loop
        import time as _time
        _q_start = _time.time()
        retrieved = agent.retrieve(
            question=q["question"],
            before_date=query_date,
            before_time=query_time,
            target_time=(target_date, target_time),
            choices={
                "A": q.get("choice_a", ""),
                "B": q.get("choice_b", ""),
                "C": q.get("choice_c", ""),
                "D": q.get("choice_d", ""),
            },
        )
        # Per-question summary: rolls up token usage + latency reported by the
        # agent's tool-use loop (Claude path). For non-tool-use backends the
        # `usage` block is absent so we fall back to wall-clock latency only.
        _q_wall = _time.time() - _q_start
        _u   = retrieved.get("usage") or {}
        _r   = _u.get("retrieval_agent") or {}
        _a   = _u.get("answering_agent") or {}
        if _u:
            # Retrieval-only breakdown.
            if _r:
                logger.info(
                    "[Q%d retrieval] rounds=%d  latency=%.2fs  in=%d  out=%d  "
                    "cache_read=%d  cache_write=%d  total=%d",
                    qid,
                    _r.get("rounds", 0), _r.get("latency_sec", 0.0),
                    _r.get("input_tokens", 0),  _r.get("output_tokens", 0),
                    _r.get("cache_read", 0),    _r.get("cache_write", 0),
                    _r.get("total_tokens", 0),
                )
            # Answering-only breakdown.
            if _a:
                logger.info(
                    "[Q%d answering] latency=%.2fs  in=%d  out=%d  "
                    "cache_read=%d  cache_write=%d  total=%d",
                    qid,
                    _a.get("latency_sec", 0.0),
                    _a.get("input_tokens", 0),  _a.get("output_tokens", 0),
                    _a.get("cache_read", 0),    _a.get("cache_write", 0),
                    _a.get("total_tokens", 0),
                )
            # Combined per-question total.
            combined_total = (_u.get("input_tokens", 0) + _u.get("output_tokens", 0)
                              + _u.get("cache_read", 0) + _u.get("cache_write", 0))
            logger.info(
                "[Q%d total] retrieval+answering  wall=%.2fs  in=%d  out=%d  "
                "cache_read=%d  cache_write=%d  total=%d",
                qid, _q_wall,
                _u.get("input_tokens", 0),  _u.get("output_tokens", 0),
                _u.get("cache_read", 0),    _u.get("cache_write", 0),
                combined_total,
            )
        else:
            logger.info("[Q%d total] wall=%.2fs  (no token-usage reported)", qid, _q_wall)

        context  = build_context(retrieved, query_date, query_time)
        prompt   = build_prompt(q, context)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ]

        # GT / Near-GT coloring helper shared by both log lines below.
        _GT_RED, _GT_BLUE, _GT_RESET = "\033[91m", "\033[94m", "\033[0m"
        _NEAR_GT_SEC = 5 * 60
        try:
            qa_target_abs = _abs_ts(target_date, target_time) \
                if target_date != "?" and target_time != "?" else None
        except Exception:
            qa_target_abs = None
        def _qa_label(c):
            base = f"{c.get('date','')} {c.get('start_time','')}-{c.get('end_time','')}"
            if qa_target_abs is None:
                return base
            try:
                cs = _abs_ts(c.get("date", "DAY1"), c.get("start_time", "0"))
                ce = _abs_ts(c.get("date", "DAY1"), c.get("end_time", "0"))
            except Exception:
                return base
            if cs <= qa_target_abs <= ce:
                return f"{_GT_RED}{base} *** GT ***{_GT_RESET}"
            if min(abs(qa_target_abs - cs), abs(qa_target_abs - ce)) <= _NEAR_GT_SEC:
                return f"{_GT_BLUE}{base} ** Next to GT **{_GT_RESET}"
            return base

        deduped_caps = retrieved.get("deduped_caps", [])
        deduped_ts = [_qa_label(c) for c in deduped_caps]
        logger.info("[QA] evidence after dedup (%d): %s",
                    len(deduped_ts), " | ".join(deduped_ts) if deduped_ts else "(none)")

        curated_caps = retrieved.get("curated_caps", [])
        curated_ts = [_qa_label(c) for c in curated_caps]
        logger.info("[QA] evidence after curation (%d): %s",
                    len(curated_ts), " | ".join(curated_ts) if curated_ts else "(none)")

        pred = (retrieved.get("answer") or "?").strip().upper()
        if pred not in ("A", "B", "C", "D"):
            pred = "?"
            logger.warning("[Q%d] answering agent returned no valid letter", qid)

        is_correct = pred == answer
        if is_correct:
            correct += 1

        # ----------------------------------------------------------------
        # Caption hit-rate per question — computed on both deduped (pre-curate)
        # and curated (post-curate) pools.
        #   HIT  — at least one caption window covers target_abs
        #   NEAR — none covers, but at least one is within ±5 min
        #   MISS — neither   N/A — no GT target_time for this question
        # ----------------------------------------------------------------
        def _pool_status(caps):
            any_hit = any_near = False
            for cap in caps:
                try:
                    cs = _abs_ts(cap.get("date") or "DAY1", cap.get("start_time") or "0")
                    ce = _abs_ts(cap.get("date") or "DAY1", cap.get("end_time")   or "0")
                except Exception:
                    continue
                if cs <= target_abs <= ce:
                    return "HIT"
                if min(abs(target_abs - cs), abs(target_abs - ce)) <= NEAR_GT_SEC:
                    any_near = True
            return "NEAR" if any_near else "MISS"

        pre_status = post_status = "N/A"
        if target_date != "?" and target_time != "?":
            try:
                target_abs = _abs_ts(target_date, target_time)
            except Exception:
                target_abs = None
            if target_abs is not None:
                gt_q_count += 1
                pre_status  = _pool_status(retrieved.get("deduped_caps", []))
                post_status = _pool_status(retrieved.get("curated_caps", []))
                if pre_status  == "HIT":  pre_hit_count  += 1
                elif pre_status  == "NEAR": pre_near_count  += 1
                if post_status == "HIT":  post_hit_count += 1
                elif post_status == "NEAR": post_near_count += 1

        cap_status = post_status  # for the results dict (backward compat)
        status = "✓" if is_correct else "✗"
        acc_so_far = correct / (i + 1) * 100
        if gt_q_count > 0:
            def _pct(n): return f"{n}/{gt_q_count}({100*n/gt_q_count:.1f}%)"
            cap_acc_str = (
                f"pre=hit:{_pct(pre_hit_count)} near:{_pct(pre_near_count)}  "
                f"post=hit:{_pct(post_hit_count)} near:{_pct(post_near_count)}"
            )
        else:
            cap_acc_str = "cap_acc=n/a"

        logger.info(
            "Q: %s | A) %s  B) %s  C) %s  D) %s",
            q["question"], q["choice_a"], q["choice_b"], q["choice_c"], q["choice_d"],
        )
        logger.info(
            "%s  pred=%s  gt=%s  [%d/%d = %.1f%%]  "
            "cap=pre:%s/post:%s  %s  (raw_caps=%d deduped=%d curated=%d)",
            status, pred, answer, correct, i + 1, acc_so_far,
            pre_status, post_status, cap_acc_str,
            len(retrieved.get("raw_captions", [])),
            len(retrieved.get("deduped_caps", [])),
            len(retrieved.get("curated_caps", [])),
        )

        results.append({
            "id": qid, "type": qtype,
            "question": q["question"],
            "pred": pred, "answer": answer, "correct": is_correct,
            "query_time": q["query_time"],
            "cap_hit": cap_status,
            "retrieved_counts": {
                "activities":   len(retrieved.get("activities", [])),
                "raw_captions": len(retrieved.get("raw_captions", [])),
            },
            "trace": retrieved.get("trace", []),
        })

    n_q = len(questions)
    if n_q == 0:
        logger.warning("No questions were evaluated — check --qids / --n-start / --n-end")
        acc = 0.0
    else:
        acc = correct / n_q * 100
    logger.info("\n=== %d/%d correct (%.1f%%) ===", correct, n_q, acc)

    by_type: dict = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        by_type[r["type"]]["total"] += 1
        if r["correct"]:
            by_type[r["type"]]["correct"] += 1
    for t, v in sorted(by_type.items()):
        pct = v["correct"] / v["total"] * 100 if v["total"] else 0
        logger.info("  %-20s %d/%d (%.1f%%)", t, v["correct"], v["total"], pct)

    out_file = config.result_file(f"{out_basename}.json", os.path.join("egor1", args.source))
    with open(out_file, "w") as f:
        json.dump({"accuracy": acc, "n": len(questions), "max_rounds": args.max_rounds, "results": results}, f, indent=2)
    logger.info("Saved → %s", out_file)


if __name__ == "__main__":
    main()
