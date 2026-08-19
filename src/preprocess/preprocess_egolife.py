"""
preprocess_egolife.py — stage 1 of preprocessing: the raw EgoLife release into
per-clip aligned caption/transcript files.

Steps
-----
1. TRANSLATE the DenseCaption SRTs (Chinese -> English), one LLM call per
   subtitle, resumable (existing translations are reused).
   -> output/preprocess/translated/{person}_{date}_{time}.jsonl
2. SYNC — align the captions and the transcripts to the video clips they
   belong to. `--source densecaption` uses the translated SRTs from step 1;
   `--source vlm` uses the video captions from caption_video.py
   (output/preprocess/vlm_video_caption/{source}/) and skips step 1 entirely.
   -> output/preprocess/{source}_sync/{person}_{date}_{time}.json

The sync files are the input to caption_egolife.py, which turns them into the
30-second caption list the memory build starts from.

Reads the EgoLife release from config.EGOLIFE_DATASET (override with
EGOCITE_EGOLIFE); writes everything under config.OUTPUT_DIR.

    python src/preprocess/preprocess_egolife.py --person A1_JAKE
    python src/preprocess/preprocess_egolife.py --person A1_JAKE --model gpt-5.4-mini
    python src/preprocess/preprocess_egolife.py --person A1_JAKE --skip-translate
"""

from __future__ import annotations

import os
import sys

# src/preprocess/<this file>  ->  src/  (so the flat `config` / `prompts` /
# `models` modules resolve when this file is run directly as a script).
_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import config  # noqa: E402
import prompts  # noqa: E402
from models import make_llm  # noqa: E402

import argparse
import glob
import json
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

import pandas as pd
import pysrt
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

TRANSLATE_SYSTEM_PROMPT = prompts.load("caption/translate")


def atomic_write_jsonl(path: str, records: List[dict]) -> None:
    parent = os.path.dirname(path) or "."
    os.makedirs(parent, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=parent) as tmp:
        tmp_path = tmp.name
        for record in records:
            json.dump(record, tmp, ensure_ascii=False)
            tmp.write("\n")
    os.replace(tmp_path, path)


def atomic_write_json(path: str, payload: dict) -> None:
    parent = os.path.dirname(path) or "."
    os.makedirs(parent, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=parent) as tmp:
        tmp_path = tmp.name
        json.dump(payload, tmp, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def format_ratio(done: int, total: int) -> str:
    if total <= 0:
        return "0/0 (0.0%)"
    return f"{done}/{total} ({done / total * 100:.1f}%)"


# ---------------------------------------------------------------------------
# Step 1 — translate DenseCaption
# ---------------------------------------------------------------------------

def _load_subtitles(srt_path: str) -> List[Tuple[int, str, str, str]]:
    subs = list(pysrt.open(srt_path))
    hour = int(os.path.basename(srt_path).split("_")[-1][:2])
    items = []
    for idx, sub in enumerate(subs, start=1):
        start = f"{(hour + sub.start.hours):02d}{sub.start.minutes:02d}{sub.start.seconds:02d}"
        end = f"{(hour + sub.end.hours):02d}{sub.end.minutes:02d}{sub.end.seconds:02d}"
        items.append((idx, sub.text, start, end))
    return items


def _load_existing_records(jsonl_path: str) -> Dict[int, dict]:
    if not os.path.exists(jsonl_path):
        return {}
    records: Dict[int, dict] = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            try:
                idx = int(record.get("custom_id", "").split("-")[0])
            except (ValueError, IndexError):
                continue
            records[idx] = record
    return records


def _collect_srt_pairs(
    input_dir: str, output_dir: str, prefix: str
) -> List[Tuple[str, str]]:
    pairs = []
    for day in sorted(d for d in os.listdir(input_dir) if d.startswith("DAY")):
        day_path = os.path.join(input_dir, day)
        if not os.path.isdir(day_path):
            continue
        for fname in sorted(f for f in os.listdir(day_path) if f.endswith(".srt") and f.startswith(prefix)):
            pairs.append((
                os.path.join(day_path, fname),
                os.path.join(output_dir, fname.replace(".srt", ".jsonl")),
            ))
    return pairs


def translate_densecap(
    data_dir: str,
    person: str,
    model,
    out_translated_dir: str,
    max_workers: int = 16,
    save_every: int = 25,
) -> None:
    input_dir = os.path.join(data_dir, "EgoLifeCap", "DenseCaption", person)
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"DenseCaption directory not found: {input_dir}")

    os.makedirs(out_translated_dir, exist_ok=True)
    prefix = f"{person}_"
    file_pairs = _collect_srt_pairs(input_dir, out_translated_dir, prefix)
    if not file_pairs:
        raise RuntimeError(f"No .srt files found under {input_dir}")

    total_segs = completed_segs = 0
    file_meta = []
    for in_file, out_file in file_pairs:
        items = _load_subtitles(in_file)
        existing = _load_existing_records(out_file)
        total_segs += len(items)
        completed_segs += len(existing)
        file_meta.append((in_file, out_file, items, existing))

    overall_bar = tqdm(
        total=total_segs, desc=f"{person} segments",
        position=0, dynamic_ncols=True, initial=completed_segs,
    )

    for file_idx, (in_file, out_file, subtitle_items, existing_records) in enumerate(file_meta, start=1):
        file_name = os.path.basename(in_file)
        name = in_file.split("/")[-3]
        date = in_file.split("/")[-2]
        file_total = len(subtitle_items)
        resumed_done = len(existing_records)

        pending = [(idx, text, start, end)
                   for idx, text, start, end in subtitle_items
                   if idx not in existing_records]

        if resumed_done >= file_total and file_total > 0:
            tqdm.write(f"[{file_idx}/{len(file_pairs)}] Skipping {file_name}: already complete")
            continue

        tqdm.write(
            f"[{file_idx}/{len(file_pairs)}] {file_name}: "
            f"{format_ratio(resumed_done, file_total)} done, {len(pending)} remaining"
        )
        file_bar = tqdm(
            total=file_total, desc=file_name,
            position=1, leave=True, dynamic_ncols=True, initial=resumed_done,
        )

        def translate_one(idx: int, text: str, start: str, end: str):
            translation = model.generate([
                {"role": "system", "content": TRANSLATE_SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ])
            record = {
                "custom_id": f"{idx}-{name}-{date}-{start}-{end}",
                "translated_text": translation,
            }
            return idx, record

        results: Dict[int, dict] = dict(existing_records)
        worker_count = min(max_workers, len(pending)) if pending else 1

        if pending:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = {
                    executor.submit(translate_one, idx, text, start, end): idx
                    for idx, text, start, end in pending
                }
                for translated_now, future in enumerate(as_completed(futures), start=1):
                    idx, record = future.result()
                    results[idx] = record
                    file_bar.update(1)
                    overall_bar.update(1)
                    completed_segs += 1
                    current_done = resumed_done + translated_now

                    if translated_now % save_every == 0 or current_done == file_total:
                        ordered = [results[i] for i in sorted(results)]
                        atomic_write_jsonl(out_file, ordered)
                        tqdm.write(
                            f"[{file_idx}/{len(file_pairs)}] {file_name} "
                            f"{format_ratio(current_done, file_total)} | overall "
                            f"{format_ratio(completed_segs, total_segs)}"
                        )

        file_bar.close()
        tqdm.write(f"Completed {file_name} -> {out_file}")

    overall_bar.close()


# ---------------------------------------------------------------------------
# Step 2 — generate sync files
# ---------------------------------------------------------------------------

def _get_captions(jsonl_path: str) -> List[Dict]:
    captions = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            cid = data["custom_id"]
            start = int(cid.split("-")[-2])
            end = int(cid.split("-")[-1])
            captions.append({"start": start, "end": end,
                              "text": data["translated_text"], "type": "caption"})
    return captions


def _get_transcripts(srt_path: str) -> List[Dict]:
    if not os.path.exists(srt_path):
        return []
    subs = list(pysrt.open(srt_path))
    hour = int(os.path.basename(srt_path).split("_")[-1][:2])
    transcripts = []
    for sub in subs:
        lines = sub.text.split("\n")
        text = lines[-1].strip() if lines else ""
        transcripts.append({
            "start": int(f"{(hour + sub.start.hours):02d}{sub.start.minutes:02d}{sub.start.seconds:02d}"),
            "end": int(f"{(hour + sub.end.hours):02d}{sub.end.minutes:02d}{sub.end.seconds:02d}"),
            "text": text,
            "type": "transcript",
        })
    return transcripts


def _find_video(files: List[str], start_time: int, end_time: int):
    for fname in files:
        try:
            video_time = int(fname.split("_")[-1][:-4])
        except ValueError:
            continue
        if start_time <= video_time < end_time:
            return fname
    return None


def _handle_time(t: int) -> int:
    if t % 10000 == 6000:
        t += 4000
    if t % 1000000 == 600000:
        t += 400000
    return t


def _match_with_video(base_dir: str, info: Tuple[str, str, str], all_df: pd.DataFrame):
    name, date, time = info
    time = int(time)
    root_dir = os.path.join(base_dir, name, date)
    if not os.path.isdir(root_dir):
        return []

    all_video_files = sorted(f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f)))
    start_time, end_time = time, time + 3000
    results = []

    while start_time < time + 1000000:
        video_file = _find_video(all_video_files, start_time, end_time)
        if video_file is not None:
            cur_df = all_df[
                (all_df["start"] >= (start_time // 100)) &
                (all_df["end"] <= (end_time // 100))
            ]
            if len(cur_df) > 0:
                results.append({
                    "video_file": video_file,
                    "data": [
                        {"start": row["start"], "end": row["end"],
                         "text": row["text"], "type": row["type"]}
                        for _, row in cur_df.iterrows()
                    ],
                })
        start_time = end_time
        end_time = _handle_time(end_time + 3000)

    return results


def _get_captions_vlm(vlm_json_path: str) -> List[Dict]:
    """Load one caption_video.py day file as sync caption rows.

    That file is {person, day, captions:[{clip, start_time, end_time, caption}]}
    with 8-digit HHMMSSFF timestamps; sync works in 6-digit HHMMSS. Empty and
    ERROR windows are dropped."""
    with open(vlm_json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    captions = []
    for c in payload.get("captions", []):
        text = (c.get("caption") or "").strip()
        if not text or text.startswith("ERROR"):
            continue
        s8, e8 = str(c["start_time"]).zfill(8), str(c["end_time"]).zfill(8)
        captions.append({
            "start": int(s8[:6]),
            "end":   int(e8[:6]),
            "text":  text,
            "type":  "caption",
        })
    return captions


def generate_sync_vlm(
    data_dir: str,
    person: str,
    vlm_caption_dir: str,
    out_sync_dir: str,
) -> None:
    """Sync for the `vlm` source: VLM video captions + transcripts -> per-hour
    sync files, in the same format and naming the densecaption path produces
    ({person}_{DAY}_{HH000000}.json). The VLM captions are per DAY, so each day
    is split into the hours it covers."""
    transcript_dir = os.path.join(data_dir, "EgoLifeCap", "Transcript", person)
    os.makedirs(out_sync_dir, exist_ok=True)

    day_files = sorted(glob.glob(os.path.join(vlm_caption_dir, f"captions_{person}_DAY*.json")))
    if not day_files:
        raise RuntimeError(f"No VLM caption files for {person} under {vlm_caption_dir}")

    n_out = 0
    for vf in tqdm(day_files, desc=f"Syncing(vlm) {person}", unit="day"):
        with open(vf, "r", encoding="utf-8") as f:
            date = json.load(f)["day"]
        caps_all = _get_captions_vlm(vf)

        for hour in sorted({c["start"] // 10000 for c in caps_all}):
            time = f"{hour:02d}000000"                  # 8-digit hour boundary
            file_stem = f"{person}_{date}_{time}"
            caps_h = [c for c in caps_all if c["start"] // 10000 == hour]
            transcripts = _get_transcripts(
                os.path.join(transcript_dir, date, f"{file_stem}.srt"))

            rows = caps_h + transcripts
            if not rows:
                continue
            all_df = pd.DataFrame(rows)
            all_df.sort_values(by="start", inplace=True)

            results = _match_with_video(data_dir, (person, date, time), all_df)
            out_path = os.path.join(out_sync_dir, f"{file_stem}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4, ensure_ascii=False)
            n_out += 1

    print(f"{n_out} sync files written to {out_sync_dir}")


def generate_sync(
    data_dir: str,
    person: str,
    out_translated_dir: str,
    out_sync_dir: str,
) -> None:
    transcript_dir = os.path.join(data_dir, "EgoLifeCap", "Transcript", person)
    os.makedirs(out_sync_dir, exist_ok=True)

    prefix = f"{person}_"
    translated_files = sorted(
        f for f in os.listdir(out_translated_dir)
        if f.endswith(".jsonl") and f.startswith(prefix)
    )
    if not translated_files:
        raise RuntimeError(f"No translated JSONL files found in {out_translated_dir}")

    for caption_file in tqdm(translated_files, desc=f"Syncing {person}"):
        file_stem = caption_file[:-6]  # strip .jsonl
        parts = file_stem.split("_")
        # file_stem: A1_JAKE_DAY1_11000000  → parts: [A1, JAKE, DAY1, 11000000]
        idx, name_part, date, time = parts[0], parts[1], parts[2], parts[3]
        name = f"{idx}_{name_part}"

        caption_path = os.path.join(out_translated_dir, caption_file)
        transcript_path = os.path.join(transcript_dir, date, f"{file_stem}.srt")

        captions = _get_captions(caption_path)
        transcripts = _get_transcripts(transcript_path)

        all_df = pd.DataFrame(captions + transcripts)
        all_df.sort_values(by="start", inplace=True)

        results = _match_with_video(data_dir, (name, date, time), all_df)
        out_path = os.path.join(out_sync_dir, f"{file_stem}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"Sync files written to {out_sync_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess EgoLife DenseCaption data: translate then sync."
    )
    parser.add_argument("--person", default="A1_JAKE",
                        help="EgoLife subject (default: A1_JAKE)")
    parser.add_argument("--data-dir", default=config.EGOLIFE_DATASET,
                        help=f"EgoLife dataset root (default: {config.EGOLIFE_DATASET})")
    parser.add_argument("--source", default="densecaption",
                        choices=config.SOURCES,
                        help="Which caption set the sync files belong to; each "
                             "writes to output/preprocess/{source}_sync/ and reads "
                             "the matching VLM backend directory "
                             f"(one of: {', '.join(config.SOURCES)}; default: densecaption)")
    parser.add_argument("--model", default="Qwen/Qwen3.6-27B-FP8",
                        help="Translation LLM. Routed by name — see "
                             "src/models/__init__.py. Default: Qwen/Qwen3.6-27B-FP8 "
                             "(local vLLM at config.VLLM_BASE_URL)")
    parser.add_argument("--max-workers", type=int, default=16,
                        help="Concurrent translation workers per file (default: 16)")
    parser.add_argument("--save-every", type=int, default=25,
                        help="Flush translated output every N segments (default: 25)")
    parser.add_argument("--skip-translate", action="store_true",
                        help="Skip translation step (reuse existing translated files)")
    parser.add_argument("--skip-sync", action="store_true",
                        help="Skip sync generation step")
    args = parser.parse_args()

    out_translated_dir = config.translated_dir(create=True)
    out_sync_dir = config.sync_dir(args.source, create=True)

    if args.source in config.VLM_SOURCES and not args.skip_translate:
        print("[Step 1] Skipped — the VLM captioner writes English directly, "
              "so there is nothing to translate.")

    if args.source not in config.VLM_SOURCES and not args.skip_translate:
        print(f"[Step 1] Translating DenseCaption for {args.person} using {args.model} ...")
        llm = make_llm(args.model)
        translate_densecap(
            data_dir=args.data_dir,
            person=args.person,
            model=llm,
            out_translated_dir=out_translated_dir,
            max_workers=args.max_workers,
            save_every=args.save_every,
        )

    if not args.skip_sync:
        print(f"[Step 2] Generating {args.source} sync files for {args.person} ...")
        if args.source in config.VLM_SOURCES:
            generate_sync_vlm(
                data_dir=args.data_dir,
                person=args.person,
                vlm_caption_dir=config.vlm_caption_dir(args.source),
                out_sync_dir=out_sync_dir,
            )
        else:
            generate_sync(
                data_dir=args.data_dir,
                person=args.person,
                out_translated_dir=out_translated_dir,
                out_sync_dir=out_sync_dir,
            )

    print("Done.")


if __name__ == "__main__":
    main()
