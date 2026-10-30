"""
Step 3b: ask a question -> find the best matching parts of the video

Usage:
    python search.py
    (type a question in English, press Enter; empty line to quit)
"""
import sys

DB_DIR = "output/chroma_db"
COLLECTION = "lecture_chunks"
MODEL_NAME = "BAAI/bge-m3"
TOP_K = 3  # how many results to show

import chromadb
from sentence_transformers import SentenceTransformer

try:
    collection = chromadb.PersistentClient(path=DB_DIR).get_collection(COLLECTION)
except Exception:
    sys.exit("Database not found. Run build_db.py first.")

print("Loading BGE-M3...")
model = SentenceTransformer(MODEL_NAME, device="cpu")


def mmss(seconds):
    seconds = int(seconds)
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


while True:
    question = input("\nQuestion (empty to quit): ").strip()
    if not question:
        break

    vector = model.encode([question], normalize_embeddings=True)[0].tolist()
    result = collection.query(query_embeddings=[vector], n_results=TOP_K)

    for rank in range(len(result["ids"][0])):
        meta = result["metadatas"][0][rank]
        distance = result["distances"][0][rank]  # smaller = closer in meaning
        text = result["documents"][0][rank]
        print(f"\n#{rank + 1}  {mmss(meta['start'])} -> {mmss(meta['end'])}"
              f"   (distance {distance:.3f})")
        print(f"   slide : {meta['slide_text'][:80] or '(none)'}")
        print(f"   image : {meta['image']}")
        print(f"   text  : {text[:200]}...")
