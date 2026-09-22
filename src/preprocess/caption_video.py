"""
caption_video.py — the `vlm` caption source: caption the video itself, with no
transcript and no dense captions.

One path only, chosen for cost and throughput:

  * FRAMES, never video. Each 15-second window is sampled at 1 fps into at most
    16 scaled JPEGs (ffmpeg, straight from the source clip — no intermediate mp4
    is ever written) and inlined as image parts. Nothing is uploaded.
  * BATCH, never per-request. Every window in a person/day goes out in batch
    jobs of `BATCH_SIZE`. Gemini submits one async batch job and polls it; the
    local vLLM server has no batch API, so a batch is fired concurrently.

Two captioners, selected by --model:
  gemini  (default)  the Gemini batch API           needs GEMINI_API_KEY
  gemma              google/gemma-4-31B-it on the   needs a vision-capable vLLM
                     local vLLM server              at config.VLLM_VISION_BASE_URL

    per day    -> output/preprocess/vlm_video_caption/{backend}/captions_{person}_{day}.json
                  (+ a .cache_*.json so an interrupted run resumes window-wise)
    per person -> output/preprocess/vlm_video_caption/{backend}/{person}_captions.json
                  (merged automatically after a person's days finish)

    {backend} is the captioner: "gemini" for any Gemini model, else the model id.

Both are INTERMEDIATES, not the final `vlm` caption set: they describe what the
video shows and nothing that was said. The final
output/captions/vlm/{person}/{person}_captions.json — what `--source vlm` feeds
into the memory build — is produced downstream, once these are aligned with the
transcripts (vlm_sync).

Needs `ffmpeg`/`ffprobe` on PATH and GEMINI_API_KEY in config.py.

    python src/preprocess/caption_video.py --person A1_JAKE             # all days, then merge
    python src/preprocess/caption_video.py --person A1_JAKE --day DAY1  # one day, then merge
"""

import argparse
import base64
import collections
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from typing import List

from pqdm.threads import pqdm
from tqdm import tqdm

# src/preprocess/<this file>  ->  src/  (so the flat `config` / `prompts`
# modules resolve when this file is run directly as a script).
_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import config  # noqa: E402
import prompts  # noqa: E402

EGOLIFE_DIR = Path(config.EGOLIFE_DATASET)
# Output is scoped by captioner, so switching models never mixes two sets of
# captions in one directory: output/preprocess/vlm_video_caption/{backend}/
WINDOW_ROOT = Path(config.preprocess_dir("vlm_video_caption"))
CONTEXT_DIR = Path(config.PROMPT_DIR) / "caption" / "egolife_vid"


def _is_local(model: str) -> bool:
    """True when this model is served by the local vLLM server rather than the
    Gemini API — currently anything Gemma."""
    return "gemma" in model.lower()


def _backend_dir(model: str) -> Path:
    """Where this model's captions live. Any Gemini id shares `gemini/`, any
    Gemma id shares `gemma/`; anything else gets a directory named after it."""
    m = model.lower()
    if "gemini" in m:
        name = "gemini"
    elif "gemma" in m:
        name = "gemma"
    else:
        name = re.sub(r"[^a-z0-9._-]+", "_", m)
    return Path(config.vlm_caption_dir(name))

MODEL      = "gemini-3-flash-preview"
# Short names accepted by --model, so the caller never types the full id.
MODEL_ALIASES = {
    "gemini": "gemini-3-flash-preview",
    "gemma":  config.VLLM_VISION_MODEL,      # google/gemma-4-31B-it via vLLM
}
BATCH_SIZE = 500        # windows per batch job
WINDOW_SEC = 15         # length of one captioned window
STRIDE_SEC = 15         # hop between windows (== WINDOW_SEC: no overlap)
FPS        = 1          # frames sampled per second inside a window
MAX_FRAMES = 16         # hard cap on images per window
POLL_SEC   = 30         # batch-status poll interval
FRAMES_ROOT = None      # when set, windows use pre-rendered annotated JPEGs
                        # from FRAMES_ROOT/<clip_stem>/ instead of ffmpeg frames


# ── person context ────────────────────────────────────────────────────────────

# which identity prompt each window resolved to, reported once per person/day
_CTX_USED: "collections.Counter" = collections.Counter()

def _load_person_context(person: str, day: str, window_sec: int = 0):
    """Return (context_text, source_filename) for this person+day+time.

    Who is on screen is the one thing the captioner cannot infer, so a window
    is NEVER captioned without a cast list. Resolution order:

      1. the timed file covering this moment: {person}_{day}_HHMMSSFF-HHMMSSFF.txt
         with start <= window_sec < end;
      2. the NEAREST timed file for that day when the moment falls in a gap
         between windows (or outside all of them) — clothing on the same day is
         the same, so the closest window is the right answer, not no answer;
      3. the plain {person}_{day}.txt;
      4. any timed file for that day, whatever its range.

    Returns ("", None) only when the day has no context file at all.
    """
    def _clean(path: Path) -> str:
        lines = [l for l in path.read_text().splitlines()
                 if l.strip() and not l.strip().startswith("#")]
        return "\n".join(lines)

    pattern = re.compile(rf"^{re.escape(person)}_{re.escape(day)}_(\d{{8}})-(\d{{8}})\.txt$")
    timed = []                       # (start, end, path)
    for ctx_path in sorted(CONTEXT_DIR.glob(f"{person}_{day}_*.txt")):
        m = pattern.match(ctx_path.name)
        if m:
            timed.append((_hhmmssff_to_sec(m.group(1)),
                          _hhmmssff_to_sec(m.group(2)), ctx_path))

    # 1. the window that covers this moment
    for t_start, t_end, ctx_path in timed:
        if t_start <= window_sec < t_end:
            return _clean(ctx_path), ctx_path.name

    # 2. otherwise the closest window on the same day (distance to its range)
    if timed:
        def _gap(w):
            s, e, _ = w
            return 0 if s <= window_sec < e else min(abs(window_sec - s),
                                                     abs(window_sec - (e - 1)))
        s, e, ctx_path = min(timed, key=_gap)
        return _clean(ctx_path), ctx_path.name

    # 3. the whole-day file
    fallback = CONTEXT_DIR / f"{person}_{day}.txt"
    if fallback.exists():
        return _clean(fallback), fallback.name

    return "", None


# ── timestamp helpers ─────────────────────────────────────────────────────────

def _hhmmssff_to_sec(ts: str) -> int:
    ts = ts.zfill(8)
    return int(ts[0:2]) * 3600 + int(ts[2:4]) * 60 + int(ts[4:6])


def _sec_to_hhmmssff(s: int) -> str:
    return f"{s // 3600:02d}{(s % 3600) // 60:02d}{s % 60:02d}00"


def _clip_start_sec(path: Path) -> int:
    m = re.search(r"_(\d{8})\.mp4$", path.name)
    return _hhmmssff_to_sec(m.group(1)) if m else 0


def _clip_duration_sec(path: Path) -> float:
    # Prefer the VIDEO STREAM duration over the container duration: some clips
    # report a longer container (e.g. 23s) than the actual video (e.g. 14.5s),
    # which would create phantom windows past the last frame. Fall back to the
    # container duration when the stream duration is unavailable.
    for entry, sel in (("stream=duration", ["-select_streams", "v:0"]),
                       ("format=duration", [])):
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", *sel, "-show_entries", entry,
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, stdin=subprocess.DEVNULL,
        )
        try:
            d = float(result.stdout.strip().splitlines()[0])
            if d > 0:
                return d
        except (ValueError, IndexError):
            continue
    return 0.0


# ── frame extraction ──────────────────────────────────────────────────────────

def _extract_frames(src: Path, win: int, win_dur: int, tmp: str):
    """Sample FPS-fps scaled JPEG frames from the SOURCE window [win, win+win_dur].

    Output-seek (-ss/-t AFTER -i) reliably yields frames even on short tails, with
    a single-frame fallback so we never get zero frames for a real window.
    Returns (frame_dir, [frame_paths]). Caller is responsible for cleanup.

    `-nostdin` + stdin=DEVNULL are load-bearing: ffmpeg otherwise takes over the
    terminal for its interactive keys and leaves it with echo disabled."""
    if FRAMES_ROOT:
        return _pick_annotated_frames(src, win, win_dur, tmp)
    frame_dir = Path(tmp) / f"{src.stem}_{win:06d}_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-i", str(src), "-ss", str(win), "-t", str(max(1, win_dur)),
         "-vf", f"fps={FPS},scale=1080:-2", "-q:v", "3", str(frame_dir / "f_%03d.jpg")],
        capture_output=True, text=True, stdin=subprocess.DEVNULL,
    )
    frames = sorted(frame_dir.glob("f_*.jpg"))[:MAX_FRAMES]
    if not frames:
        subprocess.run(
            ["ffmpeg", "-nostdin", "-y", "-ss", str(win), "-i", str(src),
             "-frames:v", "1", "-vf", "scale=1080:-2", "-q:v", "3",
             str(frame_dir / "f_001.jpg")],
            capture_output=True, text=True, stdin=subprocess.DEVNULL,
        )
        frames = sorted(frame_dir.glob("f_*.jpg"))
    return frame_dir, frames


def _pick_annotated_frames(src: Path, win: int, win_dur: int, tmp: str):
    """FRAMES_ROOT mode: this window's frames are the pre-rendered annotated
    JPEGs under FRAMES_ROOT/<clip_stem>/, chosen by the timestamps in that
    folder's detections.json, resized to 1080p so the VLM input matches the
    ffmpeg flow. A missing clip folder yields no frames — the window is
    reported as ERROR and stays uncached."""
    from PIL import Image
    frame_dir = Path(tmp) / f"{src.stem}_{win:06d}_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    det = Path(FRAMES_ROOT) / src.stem / "detections.json"
    if not det.exists():
        return frame_dir, []
    times = [r["t"] for r in json.loads(det.read_text())["frames"]]
    picked = [i for i, t in enumerate(times) if win <= t < win + win_dur][:MAX_FRAMES]
    frames = []
    for n, i in enumerate(picked, 1):
        jpg = Path(FRAMES_ROOT) / src.stem / f"f_{i:04d}.jpg"
        if not jpg.exists():
            continue
        img = Image.open(jpg)
        if img.height != 1080:
            img = img.resize((round(img.width * 1080 / img.height), 1080))
        target = frame_dir / f"f_{n:03d}.jpg"
        img.save(target, quality=90)
        frames.append(target)
    return frame_dir, frames


def _cleanup_frames(frame_dir: Path, frames: list):
    for fp in frames:
        try:
            fp.unlink()
        except Exception:
            pass
    try:
        frame_dir.rmdir()
    except Exception:
        pass


# ── prompts ───────────────────────────────────────────────────────────────────

def _system_instruction(person_context: str = "") -> str:
    instruction = prompts.load("caption/vlm_system")
    if person_context:
        instruction += "\n\n# Person Identification\n\n" + person_context
    return instruction


def _user_prompt() -> str:
    return "Watch the video and produce a first-person caption describing what you see."


# ── batch captioning ──────────────────────────────────────────────────────────

def _extract_batch_frames(batch_args: List[tuple]):
    """1-fps frames for every window in the batch (parallel — ffmpeg + disk).
    Returns (frame_dirs, frames_per_window), both aligned to batch_args."""
    dirs = [None] * len(batch_args)
    frames_for_idx = [None] * len(batch_args)

    def _ex(i):
        src, win, win_dur, _base_sec, _clip_name, _ctx, tmp = batch_args[i]
        return (i, *_extract_frames(src, win, win_dur, tmp))

    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(_ex, i) for i in range(len(batch_args))]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="  frames", leave=False):
            i, fd, frames = fut.result()
            dirs[i], frames_for_idx[i] = fd, frames
    return dirs, frames_for_idx


def _caption_batch_vllm(batch_args: List[tuple], model: str) -> List[str]:
    """Caption a batch on the local vLLM server (google/gemma-4-31B-it).

    vLLM exposes no async batch API, so the batch is fired as concurrent
    chat.completions calls with the same 1-fps frames inlined as data URIs."""
    from openai import OpenAI

    client = OpenAI(base_url=config.VLLM_VISION_BASE_URL,
                    api_key=config.VLLM_API_KEY or "EMPTY")
    captions: List[str] = [""] * len(batch_args)
    dirs, frames_for_idx = _extract_batch_frames(batch_args)

    def _one(i):
        frames = frames_for_idx[i]
        if not frames:
            return i, "ERROR: no frames"
        person_context = batch_args[i][5]
        content = [{"type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,"
                                         + base64.b64encode(fp.read_bytes()).decode()}}
                   for fp in frames]
        content.append({"type": "text", "text": _user_prompt()})
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": _system_instruction(person_context)},
                          {"role": "user",   "content": content}],
                max_tokens=512, temperature=1.0, top_p=0.95,
            )
            return i, (resp.choices[0].message.content or "").strip()
        except Exception as exc:
            return i, f"ERROR: {type(exc).__name__}: {exc}"

    todo = [i for i, f in enumerate(frames_for_idx) if f]
    print(f"  {len(todo)} windows → {sum(len(f) for f in frames_for_idx if f)} frames "
          f"→ {model} at {config.VLLM_VISION_BASE_URL}", flush=True)
    try:
        with ThreadPoolExecutor(max_workers=8) as ex:
            futs = [ex.submit(_one, i) for i in range(len(batch_args))]
            for fut in tqdm(as_completed(futs), total=len(futs), desc="  caption", leave=False):
                i, cap = fut.result()
                captions[i] = cap
    finally:
        for fd, fr in zip(dirs, frames_for_idx):
            if fd is not None:
                _cleanup_frames(fd, fr or [])
    return captions


def _caption_batch(batch_args: List[tuple], model: str) -> List[str]:
    """Caption one batch of windows. Dispatches to the local vLLM server for
    Gemma, else to the Gemini batch API."""
    if _is_local(model):
        return _caption_batch_vllm(batch_args, model)

    from google import genai as _genai

    gclient = _genai.Client(api_key=config.require("GEMINI_API_KEY"))
    captions: List[str] = [""] * len(batch_args)
    dirs, frames_for_idx = _extract_batch_frames(batch_args)

    # one inline request per window: N image parts + the text prompt
    inline_requests, req_indices = [], []
    for i, frames in enumerate(frames_for_idx):
        if not frames:
            captions[i] = "ERROR: no frames"
            continue
        person_context = batch_args[i][5]
        parts = [{"inline_data": {"mime_type": "image/jpeg",
                                  "data": base64.b64encode(fp.read_bytes()).decode()}}
                 for fp in frames]
        parts.append({"text": _user_prompt()})
        inline_requests.append({
            "contents": [{"role": "user", "parts": parts}],
            "config": {"system_instruction": _system_instruction(person_context)},
        })
        req_indices.append(i)

    if not inline_requests:
        for fd, fr in zip(dirs, frames_for_idx):
            if fd is not None:
                _cleanup_frames(fd, fr or [])
        return captions

    total_imgs = sum(len(f) for f in frames_for_idx if f)
    print(f"  {len(inline_requests)} windows → {total_imgs} inline images → batch",
          flush=True)

    try:
        job = gclient.batches.create(
            model=model,
            src=inline_requests,
            config={"display_name": f"vlm-captions-{int(time.time())}"},
        )
        print(f"  batch submitted: {job.name} ({len(inline_requests)} requests)", flush=True)

        terminal = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED",
                    "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}
        while job.state.name not in terminal:
            time.sleep(POLL_SEC)
            job = gclient.batches.get(name=job.name)
            print(f"  batch {job.name}: {job.state.name}", flush=True)

        if job.state.name != "JOB_STATE_SUCCEEDED":
            raise RuntimeError(f"batch failed: {job.state.name}")

        for idx, item in zip(req_indices, job.dest.inlined_responses):
            captions[idx] = ((item.response.text or "").strip() if item.response
                             else f"ERROR: {item.error}")
    finally:
        for fd, fr in zip(dirs, frames_for_idx):
            if fd is not None:
                _cleanup_frames(fd, fr or [])

    return captions


# ── window collection ─────────────────────────────────────────────────────────

def _collect_windows(clip: Path, person: str, day: str,
                     range_start: int, range_end: int, tmp: str) -> List[tuple]:
    base_sec = _clip_start_sec(clip)
    duration = _clip_duration_sec(clip)

    if base_sec >= range_end or base_sec + duration <= range_start:
        return []

    windows = []
    dur_int = int(duration)
    win = 0
    while win < dur_int:
        win_dur   = min(WINDOW_SEC, dur_int - win)
        abs_start = base_sec + win
        abs_end   = abs_start + win_dur
        if not (abs_start >= range_end or abs_end <= range_start):
            ctx, ctx_file = _load_person_context(person, day, abs_start)
            _CTX_USED[ctx_file or "(none — no context file for this day)"] += 1
            windows.append((clip, win, win_dur, base_sec, clip.name, ctx, tmp))
        win += STRIDE_SEC
    return windows


# ── window-level cache ────────────────────────────────────────────────────────

def _cache_path(out_dir: Path, person: str, day: str, suffix: str) -> Path:
    return out_dir / f".cache_{person}_{day}{suffix}.json"


def _cache_key(clip_name: str, start_time: str) -> str:
    return f"{clip_name}:{start_time}"


def _load_cache(cache_file: Path) -> dict:
    if cache_file.exists():
        try:
            with open(cache_file) as f:
                data = json.load(f)
            print(f"  Loaded cache: {len(data)} entries from {cache_file.name}", flush=True)
            return data
        except Exception as e:
            print(f"  Warning: could not read cache {cache_file.name}: {e}", flush=True)
    return {}


def _save_cache(cache_file: Path, cache: dict):
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(cache, f, ensure_ascii=False)


def _entry_cache_key(entry: dict) -> str:
    return _cache_key(entry["clip"], entry["start_time"])


def _args_cache_key(args: tuple) -> str:
    _src, win, _win_dur, base_sec, clip_name, _ctx, _tmp = args
    return _cache_key(clip_name, _sec_to_hhmmssff(base_sec + win))


# ── per-person/day driver ─────────────────────────────────────────────────────

def process(person: str, day: str, out_dir: Path,
            range_start: int = 0, range_end: int = 86400,
            overwrite: bool = False, model: str = MODEL,
            batch_size: int = BATCH_SIZE):
    clip_dir = EGOLIFE_DIR / person / day
    if not clip_dir.exists():
        print(f"Not found: {clip_dir}")
        return

    clips = sorted(clip_dir.glob("*.mp4"))
    if not clips:
        print(f"No clips in {clip_dir}")
        return

    suffix = ""
    if range_start > 0 or range_end < 86400:
        suffix = f"_{_sec_to_hhmmssff(range_start)}-{_sec_to_hhmmssff(range_end)}"
    out_path = out_dir / f"captions_{person}_{day}{suffix}.json"
    if out_path.exists() and not overwrite:
        print(f"Skip {person}/{day} — already done ({out_path.name})")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    cache_file = _cache_path(out_dir, person, day, suffix)
    cache = _load_cache(cache_file)

    range_str = (f"  (time range {_sec_to_hhmmssff(range_start)}–{_sec_to_hhmmssff(range_end)})"
                 if suffix else "")
    print(f"\n{person}/{day}{range_str}: {len(clips)} clips", flush=True)

    with tempfile.TemporaryDirectory() as tmp:
        _CTX_USED.clear()
        collect = partial(_collect_windows, person=person, day=day,
                          range_start=range_start, range_end=range_end, tmp=tmp)
        per_clip = pqdm(clips, collect, n_jobs=16, desc=f"prep {person}/{day}")
        all_args = [arg for r in per_clip if isinstance(r, list) for arg in r]

        # which identity prompt every window resolved to — a window captioned
        # without one produces invented names, so this is worth seeing.
        for fname, n in sorted(_CTX_USED.items(), key=lambda kv: -kv[1]):
            print(f"  identity prompt: {fname}  ({n} windows)", flush=True)

        uncached = [a for a in all_args if _args_cache_key(a) not in cache]
        print(f"  {len(all_args)} windows total "
              f"({len(all_args) - len(uncached)} cached, {len(uncached)} to process) "
              f"→ model={model} window={WINDOW_SEC}s stride={STRIDE_SEC}s "
              f"fps={FPS} max_frames={MAX_FRAMES}", flush=True)

        n_batches = (len(uncached) + batch_size - 1) // batch_size
        for b in range(n_batches):
            batch = uncached[b * batch_size:(b + 1) * batch_size]
            print(f"  Batch {b + 1}/{n_batches} ({len(batch)} windows)", flush=True)
            n_err = 0
            for args, caption in zip(batch, _caption_batch(batch, model)):
                _src, win, win_dur, base_sec, clip_name, _ctx, _tmp = args
                abs_start = base_sec + win
                entry = {
                    "clip":       clip_name,
                    "start_time": _sec_to_hhmmssff(abs_start),
                    "end_time":   _sec_to_hhmmssff(abs_start + win_dur),
                    "caption":    caption,
                }
                # Failures are NOT cached: a dead endpoint or a transient API
                # error would otherwise be baked in, skipped on every re-run,
                # and then silently dropped by merge_person — losing the window.
                # Leaving them uncached means re-running picks them up.
                if caption.startswith("ERROR:"):
                    n_err += 1
                    continue
                cache[_entry_cache_key(entry)] = entry
            _save_cache(cache_file, cache)
            print(f"  Cache saved ({len(cache)} entries total"
                  + (f", {n_err} failed and left uncached for a re-run" if n_err else "")
                  + ")", flush=True)

    all_captions = [cache[k] for k in (_args_cache_key(a) for a in all_args) if k in cache]
    missing = len(all_args) - len(all_captions)
    with open(out_path, "w") as f:
        json.dump({
            "person":    person,
            "day":       day,
            "model":     model,
            "n_windows": len(all_captions),
            "captions":  all_captions,
        }, f, indent=2, ensure_ascii=False)
    print(f"  → {len(all_captions)} captions saved to {out_path}"
          + (f"  ({missing} windows still missing — re-run to fill them)" if missing else ""))


# ── merge: per-day window files -> the per-person caption list ────────────────

def merge_person(person: str, out_dir: Path) -> str:
    """Concatenate captions_{person}_DAY*.json into one per-person list, in the
    same schema the rest of the pipeline uses:
        {start_time, end_time, text, date, video_path}
    Stays in the preprocess directory — these are video-only captions, not the
    final `vlm` caption set. Returns the path written."""
    day_files = sorted(out_dir.glob(f"captions_{person}_DAY*.json"),
                       key=lambda p: (int(re.search(r"DAY(\d+)", p.name).group(1)), p.name))
    if not day_files:
        raise FileNotFoundError(f"no captions_{person}_DAY*.json under {out_dir}")

    merged = []
    for f in day_files:
        payload = json.loads(f.read_text())
        day = payload.get("day") or re.search(r"(DAY\d+)", f.name).group(1)
        for c in payload.get("captions", []):
            text = (c.get("caption") or c.get("text") or "").strip()
            if not text or text.startswith("ERROR:"):
                continue
            clip = c.get("clip", "")
            merged.append({
                "start_time": c["start_time"],
                "end_time":   c["end_time"],
                "text":       text,
                "date":       day,
                "video_path": str(EGOLIFE_DIR / person / day / clip) if clip else "",
            })
    merged.sort(key=lambda e: (int(e["date"].replace("DAY", "")), e["start_time"]))

    out_path = out_dir / f"{person}_captions.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=4, ensure_ascii=False)
    print(f"  merged {len(day_files)} day files → {len(merged)} captions → {out_path}")
    return out_path


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    global WINDOW_SEC, STRIDE_SEC, FRAMES_ROOT
    ap = argparse.ArgumentParser(
        description="Caption EgoLife video with 1-fps frames, in batch.")
    ap.add_argument("--person",     default=None, help="e.g. A1_JAKE (omit for all)")
    ap.add_argument("--day",        default=None, help="e.g. DAY1 (omit for all)")
    ap.add_argument("--start",      default=None, help="Start time HHMMSSFF e.g. 12000000")
    ap.add_argument("--end",        default=None, help="End time HHMMSSFF e.g. 13000000")
    ap.add_argument("--model", default=MODEL,
                    help=f"Captioner. Aliases: {' | '.join(MODEL_ALIASES)} "
                         f"(gemini = {MODEL_ALIASES['gemini']}, "
                         f"gemma = {MODEL_ALIASES['gemma']} on the local vLLM "
                         f"server). Default: {MODEL}")
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                    help=f"Windows per batch job (default: {BATCH_SIZE})")
    ap.add_argument("--window-sec", type=int, default=WINDOW_SEC,
                    help=f"Caption window length in seconds (default: {WINDOW_SEC})")
    ap.add_argument("--stride-sec", type=int, default=None,
                    help="Hop between windows (default: --window-sec)")
    ap.add_argument("--overwrite",  action="store_true",
                    help="Overwrite existing output files instead of skipping")
    ap.add_argument("--frames-root", default=None,
                    help="Use pre-rendered annotated JPEGs from this root "
                         "(<root>/<clip_stem>/f_XXXX.jpg + detections.json) "
                         "instead of extracting frames with ffmpeg")
    ap.add_argument("--out-name", default=None,
                    help="Override the output directory name, e.g. gemma_w10_yolo")
    args = ap.parse_args()

    range_start = _hhmmssff_to_sec(args.start) if args.start else 0
    range_end   = _hhmmssff_to_sec(args.end)   if args.end   else 86400

    WINDOW_SEC = args.window_sec
    STRIDE_SEC = args.stride_sec if args.stride_sec is not None else args.window_sec

    args.model = MODEL_ALIASES.get(args.model.lower(), args.model)
    out_dir = _backend_dir(args.model)
    if WINDOW_SEC != 15:               # never mix window lengths in one dir
        out_dir = Path(f"{out_dir}_w{WINDOW_SEC}")
    if args.frames_root:
        FRAMES_ROOT = args.frames_root
    if args.out_name:
        out_dir = Path(config.vlm_caption_dir(args.out_name))
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"model  : {args.model}")
    print(f"output : {out_dir}")

    persons = (
        [args.person] if args.person
        else sorted(d.name for d in EGOLIFE_DIR.iterdir()
                    if d.is_dir() and re.match(r"A\d+_", d.name))
    )
    for person in persons:
        days = (
            [args.day] if args.day
            else sorted(d.name for d in (EGOLIFE_DIR / person).iterdir()
                        if d.is_dir() and re.match(r"DAY\d+", d.name))
        )
        for day in days:
            process(person, day, out_dir, range_start, range_end,
                    args.overwrite, args.model, args.batch_size)
        try:
            merge_person(person, out_dir)
        except FileNotFoundError as e:
            print(f"  skip merge for {person}: {e}")


if __name__ == "__main__":
    main()
