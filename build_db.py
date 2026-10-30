"""
Step 3: chunks.jsonl -> vectors (BGE-M3) -> Chroma database

Usage:
    python build_db.py

Input : output/chunks.jsonl
Output: output/chroma_db/   (Chroma database, stored with SQLite inside)
"""
import sys
import json
from pathlib import Path

CHUNKS_FILE = Path("output/chunks.jsonl")
DB_DIR = "output/chroma_db"
COLLECTION = "lecture_chunks"
MODEL_NAME = "BAAI/bge-m3"

if not CHUNKS_FILE.exists():
    sys.exit("output/chunks.jsonl not found. Run build_chunks.py first.")

# 1. Load chunks
chunks = []
with open(CHUNKS_FILE, encoding="utf-8") as f:
    for line in f:
        if line.strip():
            chunks.append(json.loads(line))
print(f"Loaded {len(chunks)} chunks")

# 2. Load the BGE-M3 model (first run downloads about 2 GB)
print("Loading BGE-M3 (first time: downloading, please wait)...")
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(MODEL_NAME, device="cpu")

# 3. Text -> vectors (normalized, so every vector has length 1)
print("Creating vectors...")
texts = [c["combined_text"] for c in chunks]
vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
print(f"Each vector has {len(vectors[0])} numbers")

# 4. Save into Chroma
import chromadb

client = chromadb.PersistentClient(path=DB_DIR)
try:
    client.delete_collection(COLLECTION)  # start fresh when re-running
except Exception:
    pass
collection = client.create_collection(COLLECTION)

collection.add(
    ids=[c["id"] for c in chunks],
    embeddings=[v.tolist() for v in vectors],
    documents=texts,
    metadatas=[{
        "video": c["video"],
        "start": float(c["start"]),
        "end": float(c["end"]),
        "slide_text": c["slide_text"],
        "image": c["image"] or "",
    } for c in chunks],
)

print(f"\nDone! Saved {collection.count()} chunks to {DB_DIR}")
print("Now run: python search.py")
