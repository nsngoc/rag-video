"""
Step 1: video -> speech text (Whisper) + on-screen text (PaddleOCR)

Usage:
    python extract.py video.mp4                  # run everything
    python extract.py video.mp4 --skip-whisper   # new screenshots + OCR, keep the old whisper.json
                                                 # (use this after changing INTERVAL)
    python extract.py video.mp4 --ocr-only       # re-run only OCR on the existing screenshots

Results are saved in the "output" folder:
    audio.wav       : audio extracted from the video
    frames/         : screenshots (one every INTERVAL seconds)
    meta.json       : remembers INTERVAL for the next steps
    whisper.json    : speech converted to text (with timestamps)
    ocr.json        : cleaned on-screen text (with timestamps)
"""
import os
import sys
import json
import shutil
import difflib
import argparse
import subprocess
from pathlib import Path

# Skip PaddleOCR's connectivity check on every run (faster startup)
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

# ------------------------------------------------------------------
# Settings you can edit
# ------------------------------------------------------------------
INTERVAL = 1          # take one screenshot every 1 second
MIN_CONFIDENCE = 0.7  # ignore text the OCR is not sure about (0 to 1)
MIN_LINE_CHARS = 4    # ignore very short lines such as "SMS", "@", "4"
FOOTER_Y = 0.94       # ignore text whose center is lower than 94% of the image
                      # height (logo / copyright line)
HEADER_Y = 0.0        # ignore text above this fraction (0 = keep everything)
BAD_WORDS = ["simplilearn", "rights reserved", "©"]  # drop lines containing these

# With 1 screenshot per second most neighbouring screenshots are identical.
# Skipping OCR on them saves a lot of time.
SKIP_SIMILAR = True
SIMILAR_THRESHOLD = 1.0  # average pixel difference (0-255); lower = stricter
# ------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("video", nargs="?", default="video.mp4")
parser.add_argument("--ocr-only", action="store_true",
                    help="reuse existing screenshots, only redo OCR")
parser.add_argument("--skip-whisper", action="store_true",
                    help="take new screenshots and run OCR, but skip audio/Whisper")
args = parser.parse_args()

OUT = Path("output")
FRAMES = OUT / "frames"
AUDIO = OUT / "audio.wav"
META = OUT / "meta.json"

take_frames = not args.ocr_only
run_whisper = not (args.ocr_only or args.skip_whisper)


def run(cmd):
    subprocess.run(cmd, check=True)


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ------------------------------------------------------------------
# 1. Prepare folders
# ------------------------------------------------------------------
OUT.mkdir(exist_ok=True)

if args.ocr_only:
    if not list(FRAMES.glob("frame_*.jpg")):
        sys.exit("No screenshots found in output/frames. "
                 "Run once without --ocr-only first.")
    if not META.exists():
        sys.exit("output/meta.json not found, so the screenshot interval is unknown. "
                 "Run: python extract.py video.mp4 --skip-whisper")
    INTERVAL = json.load(open(META))["interval"]  # use the interval of the saved screenshots
    print(f"Using the saved screenshots (one every {INTERVAL} second(s))")
else:
    if not Path(args.video).exists():
        sys.exit(f"Video file not found: {args.video}")
    if FRAMES.exists():
        shutil.rmtree(FRAMES)  # remove old screenshots when re-running
    FRAMES.mkdir()

# ------------------------------------------------------------------
# 2. Extract audio and screenshots with FFmpeg
# ------------------------------------------------------------------
if take_frames:
    print(f"[1/4] Extracting screenshots (one every {INTERVAL} second(s))...")
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", args.video,
         "-vf", f"fps=1/{INTERVAL}", str(FRAMES / "frame_%04d.jpg")])
    save_json({"interval": INTERVAL}, META)
if run_whisper:
    print("      Extracting audio...")
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", args.video,
         "-vn", "-ac", "1", "-ar", "16000", str(AUDIO)])

# ------------------------------------------------------------------
# 3. Whisper: speech -> text
# ------------------------------------------------------------------
if run_whisper:
    print("[2/4] Whisper is listening (CPU only, this may take a few minutes)...")
    from faster_whisper import WhisperModel

    model = WhisperModel("small.en", device="cpu", compute_type="int8")
    segments, info = model.transcribe(str(AUDIO), language="en", vad_filter=True)

    whisper_result = []
    for seg in segments:
        item = {
            "start": round(seg.start, 1),
            "end": round(seg.end, 1),
            "text": seg.text.strip(),
        }
        whisper_result.append(item)
        print(f"   {item['start']:7.1f} -> {item['end']:7.1f} | {item['text']}")

    save_json(whisper_result, OUT / "whisper.json")
    print(f"   Done: {len(whisper_result)} sentences. Saved to output/whisper.json")
else:
    print("[2/4] Skipped Whisper (keeping the existing output/whisper.json)")

# ------------------------------------------------------------------
# 4. PaddleOCR: on-screen text -> text
# ------------------------------------------------------------------
print("[3/4] PaddleOCR is reading text from the screenshots...")
from PIL import Image, ImageChops, ImageStat
from paddleocr import PaddleOCR

ocr = PaddleOCR(
    lang="en",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    enable_mkldnn=False,  # disabled to avoid errors on some Windows machines
)


def center_y(res, i):
    """Vertical center (in pixels) of the i-th text box, or None if unknown."""
    try:
        box = res["rec_boxes"][i]  # [x_min, y_min, x_max, y_max]
        return (float(box[1]) + float(box[3])) / 2
    except Exception:
        pass
    try:
        poly = res["rec_polys"][i]  # 4 corner points
        return sum(float(p[1]) for p in poly) / len(poly)
    except Exception:
        return None


def read_text(image_path):
    """Read one image and return its cleaned text."""
    height = Image.open(image_path).size[1]
    lines = []
    for res in ocr.predict(str(image_path)):
        for i, (text, score) in enumerate(zip(res["rec_texts"], res["rec_scores"])):
            text = text.strip()
            if score < MIN_CONFIDENCE or len(text) < MIN_LINE_CHARS:
                continue  # unsure or too short
            if any(w in text.lower() for w in BAD_WORDS):
                continue  # logo / copyright text
            y = center_y(res, i)
            if y is not None:
                ratio = y / height
                if ratio > FOOTER_Y or ratio < HEADER_Y:
                    continue  # text inside the logo area
            lines.append(text)
    return " ".join(lines)


def thumbnail(path):
    """Tiny grayscale copy of an image, used to detect that nothing changed."""
    return Image.open(path).convert("L").resize((64, 36))


def difference(a, b):
    """Average pixel difference between two thumbnails (0 = identical)."""
    return ImageStat.Stat(ImageChops.difference(a, b)).mean[0]


def same_slide(a, b):
    """True if two texts come from the same slide."""
    if a in b or b in a:  # text appearing step by step on one slide
        return True
    return difflib.SequenceMatcher(None, a, b).ratio() >= 0.85


ocr_result = []
frame_files = sorted(FRAMES.glob("frame_*.jpg"))
last_thumb = None  # thumbnail of the last screenshot that was really read
last_text = ""
ocr_runs = 0

for i, path in enumerate(frame_files):
    t = i * INTERVAL
    thumb = thumbnail(path)

    if SKIP_SIMILAR and last_thumb is not None and difference(thumb, last_thumb) < SIMILAR_THRESHOLD:
        text = last_text  # nothing changed: reuse the previous result
    else:
        text = read_text(path)
        last_thumb, last_text = thumb, text
        ocr_runs += 1
        print(f"   second {t:5d}: {text[:70]}")

    if len(text) < 3:  # no text in this screenshot (e.g. picture-only slide)
        continue

    # Same slide as the previous screenshot: extend its end time instead of
    # adding a new block (keep the longer text if more text appeared)
    if ocr_result and same_slide(ocr_result[-1]["text"], text):
        ocr_result[-1]["end"] = float(t + INTERVAL)
        if len(text) > len(ocr_result[-1]["text"]):
            ocr_result[-1]["text"] = text
    else:
        ocr_result.append({
            "start": float(t),
            "end": float(t + INTERVAL),
            "text": text,
        })

save_json(ocr_result, OUT / "ocr.json")
print(f"   OCR was run on {ocr_runs} of {len(frame_files)} screenshots "
      f"(the rest were unchanged).")
print(f"   Done: {len(ocr_result)} text blocks. Saved to output/ocr.json")

# ------------------------------------------------------------------
# 5. Summary
# ------------------------------------------------------------------
print("[4/4] Finished!")
print("Open these files in the output folder to see the results:")
print("  - output/whisper.json  (speech)")
print("  - output/ocr.json      (on-screen text)")
