import json
import re
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
KB_PATH = BASE_DIR / "kb.json"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

app = FastAPI(title="RAG Colosseum API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_kb():
    data = json.loads(KB_PATH.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("passages", data.get("data", []))
    items = []
    for i, item in enumerate(data):
        if isinstance(item, str):
            pid, content = i, item
        else:
            pid = int(item.get("id", item.get("index", i)))
            content = str(item.get("text", item.get("content", "")))
        if content.strip():
            items.append({"id": pid, "text": content.strip()})
    if len(items) != 50:
        raise RuntimeError(f"Expected exactly 50 passages; loaded {len(items)}")
    ids = [x["id"] for x in items]
    if sorted(ids) != list(range(50)):
        raise RuntimeError("Passage IDs must be exactly 0 through 49")
    return sorted(items, key=lambda x: x["id"])

passages = load_kb()
model = SentenceTransformer(MODEL_NAME)

# Precompute passage and sentence embeddings once when the Space starts.
passage_texts = [p["text"] for p in passages]
passage_embeddings = model.encode(
    passage_texts, normalize_embeddings=True,
    convert_to_numpy=True, show_progress_bar=False
)

sentence_records = []
for p in passages:
    for sentence in re.split(r"(?<=[.!?])\s+", p["text"]):
        sentence = sentence.strip()
        if sentence:
            sentence_records.append({
                "id": p["id"],
                "sentence": sentence,
                "passage": p["text"]
            })

sentence_embeddings = model.encode(
    [x["sentence"] for x in sentence_records],
    normalize_embeddings=True,
    convert_to_numpy=True,
    show_progress_bar=False
)

class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=3, ge=1, le=10)

def tokenize(text: str):
    return set(re.findall(r"[a-z0-9]+", text.lower()))

def retrieve(question: str, top_k: int):
    qvec = model.encode(
        [question], normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=False
    )[0]
    p_scores = passage_embeddings @ qvec
    order = np.argsort(-p_scores)[:top_k]
    selected = [
        {"id": passages[int(i)]["id"],
         "text": passages[int(i)]["text"],
         "score": float(p_scores[int(i)])}
        for i in order
    ]
    s_scores = sentence_embeddings @ qvec
    s_order = np.argsort(-s_scores)
    q_terms = tokenize(question)
    candidates = []
    for idx in s_order[:30]:
        rec = sentence_records[int(idx)]
        overlap = len(q_terms & tokenize(rec["sentence"]))
        candidates.append({
            **rec,
            "similarity": float(s_scores[int(idx)]),
            "overlap": overlap
        })
    return selected, candidates

def answer_from_evidence(question: str, selected, candidates):
    # This is deliberately extractive: retrieved document text is treated as
    # untrusted data and is never executed as instructions.
    if not candidates:
        return None, [], 0.0

    best = candidates[0]
    # Require a meaningful semantic match. Lexical overlap can support
    # paraphrases, while the semantic threshold supports synonymous wording.
    if best["similarity"] < 0.32:
        return None, [], 0.1
    if best["similarity"] < 0.40 and best["overlap"] == 0:
        return None, [], 0.1

    chosen = []
    seen = set()
    for item in candidates:
        if item["similarity"] < max(0.29, best["similarity"] - 0.13):
            continue
        if item["id"] not in {p["id"] for p in selected}:
            continue
        sentence = item["sentence"]
        if sentence not in seen:
            chosen.append(item)
            seen.add(sentence)
        if len(chosen) == 3:
            break

    if not chosen:
        return None, [], 0.1

    # Avoid returning unrelated sentences: use the best evidence sentence,
    # plus a second sentence only when it contributes distinct evidence.
    chosen.sort(key=lambda x: x["similarity"], reverse=True)
    answer = " ".join(x["sentence"] for x in chosen)
    used_ids = []
    for x in chosen:
        if x["id"] not in used_ids:
            used_ids.append(x["id"])

    confidence = float(min(0.99, max(0.61, best["similarity"])))
    return answer, used_ids, confidence

@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "RAG Colosseum",
        "passages_loaded": len(passages),
        "query_endpoint": "/query"
    }

@app.get("/health")
def health():
    return {"status": "healthy", "passages_loaded": len(passages)}

@app.post("/query")
def query(request: QueryRequest):
    question = request.question.strip()
    if not question:
        return {"answer": None, "sources": [], "scores": [], "confidence": 0.1}

    selected, candidates = retrieve(question, request.top_k)
    answer, used_ids, confidence = answer_from_evidence(
        question, selected, candidates
    )

    if answer is None:
        return {"answer": None, "sources": [], "scores": [], "confidence": 0.1}

    # Only cite passages that contributed an extracted answer sentence.
    score_by_id = {p["id"]: p["score"] for p in selected}
    ordered_ids = sorted(used_ids, key=lambda pid: score_by_id.get(pid, 0), reverse=True)
    scores = [round(float(score_by_id[pid]), 4) for pid in ordered_ids]

    # Similarity values are cosine scores. Clamp to the contract's [0, 1].
    scores = [round(max(0.0, min(1.0, x)), 4) for x in scores]
    return {
        "answer": answer,
        "sources": ordered_ids,
        "scores": scores,
        "confidence": round(max(0.0, min(1.0, confidence)), 4)
    }
