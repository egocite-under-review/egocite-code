"""
caption_egolife.py — stage 2 of preprocessing: sync files into the 30-second
first-person caption list the memory build starts from.

Reads the sync files written by preprocess_egolife.py, asks the LLM to rewrite
each video segment as one concise first-person caption (prompt:
prompt/caption/egolife_caption.txt), and writes a single JSON per person.

    input : output/preprocess/{source}_sync/{person}_*.json
    output: output/captions/{source}/{person}/{person}_captions.json

    python src/preprocess/caption_egolife.py --person A1_JAKE
    python src/preprocess/caption_egolife.py --person A1_JAKE --model gpt-5.4-mini
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
import re
from typing import Dict, List, Tuple

from pqdm.threads import pqdm as thread_pqdm
from tqdm import tqdm



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_speaker_name(person: str) -> str:
    _, _, speaker = person.partition("_")
    speaker = speaker or person
    return speaker.replace("_", " ").title()


def build_system_prompt(person: str) -> str:
    return prompts.render("caption/egolife_caption", speaker=get_speaker_name(person))


def load_sync_files(sync_dir: str, person: str) -> List[str]:
    pattern = os.path.join(sync_dir, f"{person}*.json")
    return sorted(glob.glob(pattern))


def extract_time_segments(sync_data: List[Dict]) -> List[Dict]:
    segments: List[Dict] = []
    for video_entry in sync_data:
        video_file = video_entry.get("video_file", "")
        data_entries = video_entry.get("data", [])
        if not data_entries:
            continue
        current_segment: List[Dict] = []
        for entry in data_entries:
            if isinstance(entry, dict) and "text" in entry:
                current_segment.append(entry)
                continue
            if current_segment:
                segments.append({"video_file": video_file, "entries": current_segment.copy()})
                current_segment = []
        if current_segment:
            segments.append({"video_file": video_file, "entries": current_segment.copy()})
    return segments


def normalize_camera_wearer_text(text: str, person: str) -> str:
    speaker_name = re.escape(get_speaker_name(person))
    text = re.sub(rf"(?<!\w){speaker_name}:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(rf"(?<!\w){speaker_name}'s(?!\w)", "my", text, flags=re.IGNORECASE)
    text = re.sub(rf"(?<!\w){speaker_name}(?!\w)", "I", text, flags=re.IGNORECASE)
    return text


def extract_time_info(segment_entries: List[Dict]) -> Tuple:
    if not segment_entries:
        return None, None
    start_times = [e.get("start") for e in segment_entries if e.get("start") is not None]
    end_times = [e.get("end") for e in segment_entries if e.get("end") is not None]
    if not start_times or not end_times:
        return None, None
    start_time = str(min(start_times)) + "00"
    end_time = str(max(end_times)) + "00"
    return start_time, end_time


def extract_date_from_filename(filename: str) -> str:
    match = re.search(r"DAY(\d+)", filename)
    return f"DAY{match.group(1)}" if match else "DAY1"


def extract_day_number(day_str: str) -> int:
    match = re.search(r"DAY(\d+)", day_str)
    return int(match.group(1)) if match else 0


def create_video_path(video_file: str, date: str, person: str) -> str:
    """Absolute path of the clip a caption came from (kept on every caption so
    the memory can point back at the video)."""
    if not video_file:
        return ""
    return os.path.join(config.EGOLIFE_DATASET, person, date, video_file)


def build_caption_entry(entries: List[Dict], video_file: str, sync_file: str,
                         caption_text: str, person: str) -> Dict:
    start_time, end_time = extract_time_info(entries)
    if not start_time or not end_time:
        return {}
    segment_date = extract_date_from_filename(sync_file)
    return {
        "start_time": start_time,
        "end_time": end_time,
        "text": caption_text,
        "date": segment_date,
        "video_path": create_video_path(video_file, segment_date, person),
    }


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def generate_caption(segment_entries: List[Dict], person: str, model) -> str:
    try:
        prompt = json.dumps(segment_entries, indent=2)
        content = model.generate([
            {"role": "system", "content": build_system_prompt(person)},
            {"role": "user", "content": prompt},
        ])
        generated_text = content.strip() if content else ""
        return normalize_camera_wearer_text(generated_text, person)
    except Exception as e:
        tqdm.write(f"      Error generating caption: {e}")
        return "Error generating caption."


def process_sync_files(sync_dir: str, output_file: str, person: str,
                        model, max_workers: int = 8,
                        overwrite: bool = False) -> None:
    if os.path.exists(output_file) and not overwrite:
        print(f"Skipping {output_file}: already exists (use --overwrite to regenerate)")
        return

    sync_files = load_sync_files(sync_dir, person)
    if not sync_files:
        print(f"No sync files found for {person} under {sync_dir}")
        return

    print(f"Found {len(sync_files)} sync files for {person}")
    all_captions = []

    for sync_file_index, sync_file in enumerate(sync_files, start=1):
        tqdm.write(f"\033[91mProcessing [{sync_file_index}/{len(sync_files)}]: {os.path.basename(sync_file)}\033[0m")
        try:
            with open(sync_file, "r", encoding="utf-8") as f:
                sync_data = json.load(f)

            segments = extract_time_segments(sync_data)
            usable_segments = [s for s in segments if s.get("entries")]
            segment_entries_list = [s["entries"] for s in usable_segments]
            segment_video_files = [s["video_file"] for s in usable_segments]

            if not segment_entries_list:
                continue

            args = [(entries, person, model) for entries in segment_entries_list]
            raw_results = thread_pqdm(
                args,
                generate_caption,
                n_jobs=min(max_workers, len(args)),
                argument_type="args",
                desc=f"  {os.path.basename(sync_file)}",
                leave=False,
            )

            for i, caption_text in enumerate(raw_results):
                if isinstance(caption_text, Exception):
                    tqdm.write(f"      Error at segment {i}: {caption_text}")
                    caption_text = "Error generating caption."
                entry = build_caption_entry(
                    entries=segment_entries_list[i],
                    video_file=segment_video_files[i],
                    sync_file=sync_file,
                    caption_text=caption_text,
                    person=person,
                )
                if entry:
                    all_captions.append(entry)
                    tqdm.write(
                        f"    {entry['video_path']}: {entry['start_time']}-{entry['end_time']}"
                    )

        except Exception as e:
            tqdm.write(f"    \033[91mError processing {sync_file}: {e}\033[0m")
            continue

    print(f"\nSorting {len(all_captions)} captions by date and start time...")
    all_captions.sort(key=lambda x: (extract_day_number(x["date"]), x["start_time"]))

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_captions, f, indent=4, ensure_ascii=False)

    print(f"Generated {len(all_captions)} captions → {output_file}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate first-person captions from EgoLife sync files."
    )
    parser.add_argument("--person", default="A1_JAKE",
                        help="Person identifier (default: A1_JAKE)")
    parser.add_argument("--model", default=None,
                        help="Captioning LLM. Routed by name — see "
                             "src/models/__init__.py. Defaults to the model that "
                             "produced the source's video captions: "
                             f"{config.VLLM_VISION_MODEL} for --source gemma, "
                             "Qwen/Qwen3.6-27B-FP8 otherwise.")
    parser.add_argument("--sync-dir", default=None,
                        help="Directory of sync JSON files "
                             "(default: output/preprocess/{source}_sync)")
    parser.add_argument("--source", default="densecaption",
                        choices=config.SOURCES,
                        help="Which caption set this run produces (default: densecaption)")
    parser.add_argument("--max-workers", type=int, default=8,
                        help="Concurrent requests (default: 8)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing output file")
    args = parser.parse_args()

    # A gemma run stays on one model end to end: the same server that captioned
    # the video also merges those captions with the transcript.
    if args.model is None:
        args.model = (config.VLLM_VISION_MODEL if args.source == "gemma"
                      else "Qwen/Qwen3.6-27B-FP8")

    sync_dir = args.sync_dir or config.sync_dir(args.source)
    output_file = config.caption_file(args.person, args.source, create=True)
    print(f"Captioning {args.person} with {args.model!r}")
    print(f"  sync   : {sync_dir}")
    print(f"  output : {output_file}")
    model = make_llm(args.model)

    process_sync_files(
        sync_dir=sync_dir,
        output_file=output_file,
        person=args.person,
        model=model,
        max_workers=args.max_workers,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
