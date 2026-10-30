"""
Step 4: question -> find parts of the video (Chroma) -> short answer (Gemini)

Full RAG loop:
    Retrieval  : BGE-M3 + Chroma find the most relevant parts of the video
    Generation : Google Gemini reads only those parts and writes a short answer

Setup (in the same CMD window, before running):
    pip install google-genai
    set GEMINI_API_KEY=your-key        (get one at https://aistudio.google.com/apikey)

Usage:
    python answer.py
    (type a question, press Enter; empty line to quit)
"""
import os
import sys

# ------------------------------------------------------------------
# Settings you can edit
# ------------------------------------------------------------------
# "-latest" is an alias that points to the current Flash-Lite model.
# If you get a 404 / "model not found" error, copy an exact model name
# from Google AI Studio and paste it here.
GEMINI_MODEL = "gemini-flash-lite-latest"

DB_DIR = "output/chroma_db"
COLLECTION = "lecture_chunks"
EMBED_MODEL = "BAAI/bge-m3"
TOP_K = 3            # how many parts of the video to read
MAX_DISTANCE = 0.95  # parts farther than this are treated as "not related"
# ------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You answer questions about a lecture video using ONLY the video parts "
    "provided by the user. Rules:\n"
    "- Give a short, direct answer (1 to 3 sentences). Do not copy long passages.\n"
    "- Mention the time of the part you used, like (00:41).\n"
    "- If the parts do not contain the answer, say you cannot find it in the "
    "video. Never guess.\n"
    "- Slide text comes from OCR: it may contain mistakes, and it does not tell "
    "which number belongs to which picture. Prefer the speech text for facts.\n"
    "- Answer in the same language as the question."
)

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    sys.exit("GEMINI_API_KEY is not set. In CMD run:  set GEMINI_API_KEY=your-key")

import chromadb
from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer

try:
    collection = chromadb.PersistentClient(path=DB_DIR).get_collection(COLLECTION)
except Exception:
    sys.exit("Database not found. Run build_db.py first.")

client = genai.Client(api_key=api_key)
print(f"Using Gemini model: {GEMINI_MODEL}")
print("Loading BGE-M3...")
embedder = SentenceTransformer(EMBED_MODEL, device="cpu")


def mmss(seconds):
    seconds = int(seconds)
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


while True:
    question = input("\nQuestion (empty to quit): ").strip()
    if not question:
        break

    # 1. Retrieval: find the closest parts of the video
    vector = embedder.encode([question], normalize_embeddings=True)[0].tolist()
    result = collection.query(query_embeddings=[vector], n_results=TOP_K)

    parts = []
    for i in range(len(result["ids"][0])):
        if result["distances"][0][i] <= MAX_DISTANCE:
            parts.append({
                "meta": result["metadatas"][0][i],
                "text": result["documents"][0][i],
                "distance": result["distances"][0][i],
            })

    if not parts:
        print("\nNo related part found in the video.")
        continue

    # 2. Generation: give those parts to Gemini and ask for a short answer
    context = "\n\n".join(
        f"[Part {n} | {mmss(p['meta']['start'])} - {mmss(p['meta']['end'])}]\n{p['text']}"
        for n, p in enumerate(parts, start=1)
    )
    user_message = f"Video parts:\n{context}\n\nQuestion: {question}"

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=user_message,
            config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
        )
        answer = response.text
    except Exception as e:
        print(f"\nGemini error: {e}")
        print("(404 = wrong model name; 400/401/403 = wrong key; 429 = free limit reached, wait a bit)")
        continue

    print(f"\nAnswer: {answer}")
    print("\nSources:")
    for p in parts:
        print(f"  {mmss(p['meta']['start'])} -> {mmss(p['meta']['end'])}"
              f"  (distance {p['distance']:.3f})  {p['meta']['image']}")
