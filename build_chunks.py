"""
Step 2: whisper.json + ocr.json -> chunks.jsonl

  1. Clean the speech text
  2. Group short sentences into chunks of about TARGET_SEC seconds (with overlap)
  3. Align each chunk with the slide text that is on screen at the same time
  4. Pick one screenshot per chunk and build the text used for vectors

Usage:
    python build_chunks.py              # video name defaults to "video"
    python build_chunks.py lecture1     # chunk ids become lecture1_001, ...

Input : output/whisper.json, output/ocr.json, output/frames/
Output: output/chunks.jsonl  (one JSON chunk per line)
"""
import re
import sys
import json
from pathlib import Path

# ------------------------------------------------------------------
# Settings you can edit
# ------------------------------------------------------------------
TARGET_SEC = 45        # close a chunk once it lasts at least this many seconds
MAX_WORDS = 180        # ...or once it reaches this many words
OVERLAP = 1            # the next chunk starts with the last sentence of the previous
MIN_WORDS = 2          # drop sentences shorter than this ("Okay.", "Um.")
MIN_OVERLAP_SEC = 3    # a slide must be on screen at least this long to be attached
# ------------------------------------------------------------------

VIDEO_NAME = sys.argv[1] if len(sys.argv) > 1 else "video"
OUT = Path("output")
FRAMES = OUT / "frames"


def load_interval():
    """Seconds between screenshots, saved by extract.py."""
    meta = OUT / "meta.json"
    if meta.exists():
        with open(meta, encoding="utf-8") as f:
            return json.load(f)["interval"]
    print("Warning: output/meta.json not found, assuming one screenshot every 5 seconds.")
    return 5


INTERVAL = load_interval()


def load_json(path):
    if not path.exists():
        sys.exit(f"File not found: {path}. Run extract.py first.")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------------
# 1. Clean
# ------------------------------------------------------------------
def clean_text(text):
    return re.sub(r"\s+", " ", text).strip()


whisper = load_json(OUT / "whisper.json")
ocr = load_json(OUT / "ocr.json")

segments = []
for s in whisper:
    text = clean_text(s["text"])
    if len(text.split()) >= MIN_WORDS:
        segments.append({"start": s["start"], "end": s["end"], "text": text})

ocr_blocks = []
for b in ocr:
    text = clean_text(b["text"])
    if len(text) >= 4:
        ocr_blocks.append({"start": b["start"], "end": b["end"], "text": text})

print(f"Speech sentences kept: {len(segments)} / {len(whisper)}")
print(f"Slide text blocks kept: {len(ocr_blocks)} / {len(ocr)}")
if not segments:
    sys.exit("No speech text found in whisper.json.")


# ------------------------------------------------------------------
# 2. Chunking: group sentences, never cut inside a sentence
# ------------------------------------------------------------------
def make_chunks(segs):
    groups = []
    i = 0
    while i < len(segs):
        j = i
        words = 0
        while j < len(segs):
            words += len(segs[j]["text"].split())
            j += 1
            duration = segs[j - 1]["end"] - segs[i]["start"]
            if duration >= TARGET_SEC or words >= MAX_WORDS:
                break
        groups.append(segs[i:j])
        if j >= len(segs):
            break
        i = max(j - OVERLAP, i + 1)  # step back by OVERLAP, but always move forward
    return groups


# ------------------------------------------------------------------
# 3. Alignment: attach slide text by time overlap
# ------------------------------------------------------------------
def time_overlap(a_start, a_end, b_start, b_end):
    return min(a_end, b_end) - max(a_start, b_start)


def frame_path(t):
    """Screenshot closest to second t (or None if there are no screenshots)."""
    files = sorted(FRAMES.glob("frame_*.jpg"))
    if not files:
        return None
    index = min(max(round(t / INTERVAL), 0), len(files) - 1)
    return files[index].as_posix()


chunks = []
for n, group in enumerate(make_chunks(segments), start=1):
    start = group[0]["start"]
    end = group[-1]["end"]
    speech_text = " ".join(s["text"] for s in group)

    matches = []
    for b in ocr_blocks:
        ov = time_overlap(start, end, b["start"], b["end"])
        if ov >= MIN_OVERLAP_SEC:
            matches.append((ov, b))
    matches.sort(key=lambda m: m[1]["start"])  # in the order they appear

    slide_text = " ; ".join(dict.fromkeys(b["text"] for _, b in matches))

    if matches:
        # main slide = the one on screen the longest; use a screenshot from its middle
        _, main = max(matches, key=lambda m: m[0])
        t = (max(start, main["start"]) + min(end, main["end"])) / 2
    else:
        t = (start + end) / 2  # picture-only part: use the middle of the chunk

    if slide_text:
        combined = f"Slide: {slide_text}. Speech: {speech_text}"
    else:
        combined = f"Speech: {speech_text}"

    chunks.append({
        "id": f"{VIDEO_NAME}_{n:03d}",
        "video": VIDEO_NAME,
        "start": round(start, 1),
        "end": round(end, 1),
        "speech_text": speech_text,
        "slide_text": slide_text,
        "image": frame_path(t),
        "combined_text": combined,
    })

# ------------------------------------------------------------------
# 4. Save as JSONL (one chunk per line)
# ------------------------------------------------------------------
out_path = OUT / "chunks.jsonl"
with open(out_path, "w", encoding="utf-8") as f:
    for c in chunks:
        f.write(json.dumps(c, ensure_ascii=False) + "\n")

# A caption backup made by add_captions.py belongs to the OLD chunks: remove it,
# otherwise add_captions.py would restore the old chunks.
old_backup = OUT / "chunks_before_captions.jsonl"
if old_backup.exists():
    old_backup.unlink()
    print("Removed the old caption backup. Run add_captions.py again if you use BLIP.")

print(f"\nCreated {len(chunks)} chunks -> {out_path}\n")
for c in chunks:
    words = len(c["speech_text"].split())
    print(f"{c['id']}  {c['start']:6.1f}s -> {c['end']:6.1f}s  {words:3d} words  "
          f"image={Path(c['image']).name if c['image'] else None}")
    print(f"    slide : {c['slide_text'][:80] or '(no slide text)'}")
    print(f"    speech: {c['speech_text'][:80]}...")
