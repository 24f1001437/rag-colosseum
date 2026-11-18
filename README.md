---
title: RAG Colosseum
emoji: 🔎
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# RAG Colosseum

FastAPI semantic retrieval service for the 50-passage assignment knowledge base.

## API

`POST /query`

Request:
```json
{
  "question": "What is the Raft heartbeat interval?",
  "top_k": 3
}
```

Response:
```json
{
  "answer": "The leader sends 150ms heartbeats to prevent follower election timeouts.",
  "sources": [0],
  "scores": [0.8],
  "confidence": 0.8
}
```

Also available:
- `GET /` service status
- `GET /health` health check
- `GET /docs` interactive API documentation

The embedding model is downloaded on first startup. Initial build/startup can take several minutes.
This implementation is extractive: it returns evidence sentences from the supplied passages and does not execute instructions found in retrieved text.
