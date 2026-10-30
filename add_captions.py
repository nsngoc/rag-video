"""
Step 2b (optional): describe each chunk's screenshot with BLIP and add the
description to the chunk text.

OCR reads WORDS on the screen; BLIP describes the PICTURE (people, objects,
diagrams). Using both gives the search more information.

Usage:
    python add_captions.py

Input : output/chunks.jsonl   (made by build_chunks.py)
Output: output/chunks.jsonl   (updated: new field "visual_text", new combined_text)
        output/chunks_before_captions.jsonl  (backup of the version without captions)

Run order:
    python build_chunks.py
    python add_captions.py
    python build_db.py
"""
import sys
import json
import shutil
from pathlib import Path

OUT = Path("output")
CHUNKS = OUT / "chunks.jsonl"
BACKUP = OUT / "chunks_before_captions.jsonl"
BLIP_MODEL = "Salesforce/blip-image-captioning-base"
MAX_NEW_TOKENS = 40


def build_combined(slide_text, visual_text, speech_text):
    """Same format as build_chunks.py, plus an optional 'Visual:' part."""
    parts = []
    if slide_text:
        parts.append(f"Slide: {slide_text}.")
    if visual_text:
        parts.append(f"Visual: {visual_text}.")
    parts.append(f"Speech: {speech_text}")
    return " ".join(parts)


def load_chunks(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


if __name__ == "__main__":
    # Always start from the version WITHOUT captions, so re-running is safe
    if BACKUP.exists():
        source = BACKUP
    elif CHUNKS.exists():
        shutil.copy(CHUNKS, BACKUP)
        source = BACKUP
    else:
        sys.exit("output/chunks.jsonl not found. Run build_chunks.py first.")

    chunks = load_chunks(source)
    print(f"Loaded {len(chunks)} chunks")

    print("Loading BLIP (first run downloads about 1 GB, please wait)...")
    import torch
    from PIL import Image
    from transformers import BlipProcessor, BlipForConditionalGeneration

    processor = BlipProcessor.from_pretrained(BLIP_MODEL)
    model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL)
    model.eval()

    cache = {}  # the same screenshot is described only once

    def describe(image_path):
        if image_path in cache:
            return cache[image_path]
        image = Image.open(image_path).convert("RGB")
        inputs = processor(image, return_tensors="pt")
        with torch.no_grad():
            ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
        text = processor.decode(ids[0], skip_special_tokens=True).strip().rstrip(".")
        cache[image_path] = text
        return text

    for c in chunks:
        path = c.get("image")
        if path and Path(path).exists():
            c["visual_text"] = describe(path)
        else:
            c["visual_text"] = ""
        c["combined_text"] = build_combined(c["slide_text"], c["visual_text"], c["speech_text"])
        print(f"{c['id']}  {c['start']:6.1f}s  visual: {c['visual_text'] or '(none)'}")

    with open(CHUNKS, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"\nDone. Updated {CHUNKS}")
    print("Now run: python build_db.py")
