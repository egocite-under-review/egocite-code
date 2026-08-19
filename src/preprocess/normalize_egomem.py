#!/usr/bin/env python3
"""
normalize_egomem.py — EgoMem (LifeDialBench) -> the EgoLifeQA convention.

Source: https://github.com/RayNeo-AI-2025/LifeDialBench
        cloned to dataset/LifeDialBench/ (see dataset/README.md)

Output: dataset/LifeDialBench/data/EgoMem-Normalized.json, which is what
        src/eval/eval_egomem.py reads (config.EGOMEM_DATASET). That file is
        DERIVED, not upstream — regenerate it only if the released data changes,
        since step 1 below calls an LLM per question.

Normalize EgoMem.json:
1. Replace history_name mentions in question/options/answer with I/me/my (via LLM)
2. Dates: 2025-08-16=DAY1 ... 2025-08-23=DAY8 in all text and structured fields
3. Times: HH:MM → HHMM0000 integer format in all text and structured fields
4. question_time: "YYYY-MM-DD" → {date: "DAY(X-1)", time: 23590000}
5. evidence_date: "YYYY-MM-DD, HH:MM-HH:MM" → {date: "DAYX", start_time: ..., end_time: ...}
"""

import json
import asyncio
import os
import sys
from openai import AsyncOpenAI

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

MODEL = "gpt-5.4"

DATE_MAP = {
    "2025-08-16": "DAY1",
    "2025-08-17": "DAY2",
    "2025-08-18": "DAY3",
    "2025-08-19": "DAY4",
    "2025-08-20": "DAY5",
    "2025-08-21": "DAY6",
    "2025-08-22": "DAY7",
    "2025-08-23": "DAY8",
}

# Reverse map: DAY1 → "2025-08-16" index
DAY_INDEX = {v: i + 1 for i, v in enumerate(DATE_MAP.values())}

SYSTEM_PROMPT_TEMPLATE = """You are normalizing QA benchmark data from the first-person perspective of a person named "{name}".

Apply ALL three transformations to the question, each option, and the answer:

1. PRONOUN REPLACEMENT
   - Replace "{name}" with first-person pronouns (I / me / my / myself / mine) based on grammatical role.
   - If an option uses a third-person pronoun (he/she/they/his/her/their/him) that refers to {name}, convert it to the matching first-person form.
   - Other people's names and their pronouns stay unchanged.

2. DATE NORMALIZATION — use this exact mapping, matching any surface form:
   2025-08-16 / August 16, 2025 / Aug 16 2025  →  DAY1
   2025-08-17 / August 17, 2025 / Aug 17 2025  →  DAY2
   2025-08-18 / August 18, 2025 / Aug 18 2025  →  DAY3
   2025-08-19 / August 19, 2025 / Aug 19 2025  →  DAY4
   2025-08-20 / August 20, 2025 / Aug 20 2025  →  DAY5
   2025-08-21 / August 21, 2025 / Aug 21 2025  →  DAY6
   2025-08-22 / August 22, 2025 / Aug 22 2025  →  DAY7
   2025-08-23 / August 23, 2025 / Aug 23 2025  →  DAY8

3. TIME NORMALIZATION — convert every time mention to an integer using format HHMM0000:
   - 24-hour "HH:MM"     → HH*1000000 + MM*10000     e.g. 20:10 → 20100000
   - 12-hour "H:MM AM"   → hour*1000000 + min*10000  e.g. 10:00 AM → 10000000
   - 12-hour "H:MM PM"   → (hour+12 if hour<12)*1000000 + min*10000
                            e.g. 7:40 PM → 19400000,  12:30 PM → 12300000
   - Ranges "HH:MM-HH:MM" → both ends converted, hyphen kept  e.g. 13:20-13:30 → 13200000-13300000
   - "from X to Y"       → both ends converted individually
   - Combined date+time "2025-08-16, 11:10-11:20" → "DAY1, 11100000-11200000"
   - "August 20, 2025, from 7:40 PM to 7:50 PM"  → "DAY5, from 19400000 to 19500000"

Return ONLY valid JSON (no markdown, no explanation):
{{"question": "...", "options": ["A. ...", "B. ...", "C. ...", "D. ..."], "answer": "..."}}"""


def time_str_to_int(h: int, m: int) -> int:
    return h * 1_000_000 + m * 10_000


def parse_hhmm(s: str) -> int:
    """Parse "HH:MM" (24h) → integer."""
    h, m = s.strip().split(":")
    return time_str_to_int(int(h), int(m))


def normalize_question_time(qt: str) -> dict:
    """
    "2025-08-17" (DAY2) → {date: "DAY1", time: 23590000}
    question_time is always the next day after the event, so we subtract 1.
    """
    day_label = DATE_MAP[qt]          # e.g. "DAY2"
    day_num = int(day_label[3:])      # 2
    prev = f"DAY{day_num - 1}"        # "DAY1"
    return {"date": prev, "time": 23590000}


def normalize_evidence_date(ed: str) -> dict:
    """
    "2025-08-19, 21:30-21:40"
    → {date: "DAY4", start_time: 21300000, end_time: 21400000}
    """
    date_part, time_part = ed.split(",", 1)
    date_part = date_part.strip()
    time_part = time_part.strip()
    day_label = DATE_MAP[date_part]
    start_str, end_str = time_part.split("-")
    return {
        "date": day_label,
        "start_time": parse_hhmm(start_str),
        "end_time": parse_hhmm(end_str),
    }


async def llm_normalize_qa(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    name: str,
    qa: dict,
) -> dict:
    """Call LLM to normalize question, options, and answer for a single QA."""
    system = SYSTEM_PROMPT_TEMPLATE.format(name=name)
    user_payload = {
        "question": qa["question"],
        "options": qa["options"],
        "answer": qa["answer"],
    }
    user_msg = json.dumps(user_payload, ensure_ascii=False)

    async with sem:
        response = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )

    raw = response.choices[0].message.content
    try:
        normalized = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"[WARN] JSON parse error for qa: {qa['question'][:60]!r}: {e}")
        normalized = user_payload  # fall back to original

    return normalized


async def process_record(client: AsyncOpenAI, sem: asyncio.Semaphore, record: dict) -> dict:
    name = record["history_name"]
    qa_list = record["qa"]

    # Build all LLM tasks for this record
    tasks = [
        llm_normalize_qa(client, sem, name, qa)
        for qa in qa_list
    ]
    llm_results = await asyncio.gather(*tasks)

    normalized_qa = []
    for qa, llm_out in zip(qa_list, llm_results):
        new_qa = {
            "question": llm_out.get("question", qa["question"]),
            "options": llm_out.get("options", qa["options"]),
            "golden_option": qa["golden_option"],
            "answer": llm_out.get("answer", qa["answer"]),
            "question_time": normalize_question_time(qa["question_time"]),
            "question_type": qa["question_type"],
            "evidence_date": normalize_evidence_date(qa["evidence_date"]),
            "timescale": qa["timescale"],
        }
        normalized_qa.append(new_qa)

    return {
        "history_name": name,
        "qa": normalized_qa,
        "chat_time": record.get("chat_time"),
        "chat_history": record.get("chat_history"),
    }


async def main():
    data_dir = os.path.join(config.EGOMEM_DATASET, "data")
    input_path = os.path.join(data_dir, "EgoMem.json")
    output_path = os.path.join(data_dir, "EgoMem-Normalized.json")

    print(f"Loading {input_path} …")
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    total_qa = sum(len(d["qa"]) for d in data)
    print(f"Records: {len(data)}, Total QAs: {total_qa}")

    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    sem = asyncio.Semaphore(20)  # max 20 concurrent LLM calls

    results = []
    for i, record in enumerate(data):
        name = record["history_name"]
        print(f"[{i+1}/{len(data)}] Processing {name} ({len(record['qa'])} QAs) …")
        normalized = await process_record(client, sem, record)
        results.append(normalized)
        print(f"  ✓ {name} done")

    print(f"\nWriting {output_path} …")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
