# 🔥 Roast My Resume — RAG Edition

> **Academic ESE Project**  
> Evidence-based resume critique powered by Hybrid RAG (Qdrant) + LangGraph + FastAPI

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Streamlit UI (app.py)                    │
│  Sidebar: PDF upload + JD text area                             │
│  Main: Score dashboard · Brutal Roast · Actionable Fixes        │
│  Expander: 🔍 RAG Retrieval Context (evaluator transparency)    │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTP POST /api/evaluate
┌────────────────────────────▼────────────────────────────────────┐
│                      FastAPI  (main.py)                         │
│  POST /api/evaluate   GET /api/health   GET /api/collection-info│
└────────────────────────────┬────────────────────────────────────┘
                             │ roast_graph.invoke(state)
┌────────────────────────────▼────────────────────────────────────┐
│               LangGraph StateGraph  (graph.py)                  │
│                                                                  │
│  ┌─────────────┐   ┌──────────────────┐   ┌──────────────────┐ │
│  │ node_parse  │──▶│  node_retrieve   │──▶│  node_generate   │ │
│  │ (LLM + JSON)│   │ (Qdrant Hybrid)  │   │  (LLM + JSON)    │ │
│  └─────────────┘   └──────────────────┘   └──────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                             │ dense + sparse search
┌────────────────────────────▼────────────────────────────────────┐
│                    Qdrant Vector DB                             │
│  Collection: roast_resume                                       │
│  Vectors: dense (OpenAI/Ollama) + sparse BM25                  │
│  Metadata filters: ats_rule | writing_rule | gold_example       │
└─────────────────────────────────────────────────────────────────┘
```

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Backend API | FastAPI + Pydantic v2 |
| AI Orchestration | LangGraph (StateGraph) |
| LLM | OpenAI GPT-4o-mini (default) / Ollama (local GPU) |
| Embeddings | OpenAI text-embedding-3-small / Ollama nomic-embed-text |
| Vector DB | Qdrant — Hybrid Search (dense + BM25 sparse) |
| Document Processing | pypdf |
| Frontend | Streamlit |

---

## Project Structure

```
LLM_ESE/
├── data/                          # Ground-truth PDFs (not committed)
│   ├── ApplicantTrackingSystem-Guide_2.pdf
│   ├── Leland—The-Ultimate-Career-Resume-Guide+Examples_removed_2.pdf
│   └── Resume-Book_removed.pdf
│
├── doc/
│   └── project_context_and_architecture_plan.md
│
├── config.py          # Centralised env-driven settings (pydantic-settings)
├── llm_factory.py     # Provider-agnostic LLM & Embedding factory
├── ingest.py          # One-time PDF ingestion into Qdrant
├── graph.py           # LangGraph state machine (3 nodes)
├── main.py            # FastAPI server
├── app.py             # Streamlit UI
│
├── requirements.txt
├── .env.example       # Copy to .env and fill in your keys
├── .gitignore
└── README.md
```

---

## Quick Start

### 1. Prerequisites

- Python 3.10+
- [Qdrant](https://qdrant.tech/documentation/quick-start/) running locally on port 6333:
  ```bash
  docker run -p 6333:6333 qdrant/qdrant
  ```
- OpenAI API key **or** Ollama running locally

### 2. Install dependencies

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure environment

```bash
copy .env.example .env
# Edit .env — set OPENAI_API_KEY (or switch LLM_PROVIDER=ollama)
```

### 4. Add the ground-truth PDFs

Copy the three PDFs into the `data/` directory:
- `ApplicantTrackingSystem-Guide_2.pdf`
- `Leland—The-Ultimate-Career-Resume-Guide+Examples_removed_2.pdf`
- `Resume-Book_removed.pdf`

### 5. Run ingestion (one time)

```bash
python ingest.py
```

This chunks, embeds, and upserts all three documents into Qdrant with metadata tags.

### 6. Start the FastAPI backend

```bash
python main.py
# or: uvicorn main:app --reload --port 8000
```

### 7. Start the Streamlit frontend

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## LangGraph Pipeline Details

### State Schema

```python
class GraphState(TypedDict):
    resume_text       : str           # Raw PDF text
    job_description   : str           # Target JD
    parsed_sections   : dict          # {work_experience, education, skills, summary}
    retrieved_context : list[dict]    # Qdrant chunks with metadata
    final_roast       : dict          # {overall_score, brutal_roast, actionable_fixes, ...}
    error             : str | None    # Set on any node failure
```

### Node 1 — `node_parse`
Uses an LLM with structured output to parse the raw PDF text into JSON with keys:
`work_experience`, `education`, `skills`, `summary`.

### Node 2 — `node_retrieve`
Runs **three targeted Hybrid Search queries** against Qdrant:
1. **ATS rules** (`doc_type=ats_rule`) — formatting & machine-readability rules
2. **Writing rules** (`doc_type=writing_rule`) — XYZ formula, action verbs, impact metrics
3. **Gold examples** (`doc_type=gold_example`) — MIT engineering resume benchmarks

Each query uses both **dense cosine similarity** (semantic) and **BM25 sparse vectors** (keyword-exact), merged and de-duplicated.

### Node 3 — `node_generate`
Injects resume, JD, and all retrieved context chunks into a strict system prompt.
Returns a JSON object:
```json
{
  "overall_score": 62,
  "brutal_roast": "...",
  "actionable_fixes": [
    {
      "issue": "...",
      "fix": "...",
      "rule_reference": "...",
      "severity": "critical|major|minor"
    }
  ],
  "ats_verdict": "PASS|FAIL",
  "jd_alignment_score": 48
}
```

---

## Switching to Ollama (Local GPU)

1. [Install Ollama](https://ollama.ai) and pull the models:
   ```bash
   ollama pull llama3.1
   ollama pull nomic-embed-text
   ```
2. In `.env`:
   ```
   LLM_PROVIDER=ollama
   OLLAMA_BASE_URL=http://localhost:11434
   OLLAMA_MODEL=llama3.1
   OLLAMA_EMBEDDING_MODEL=nomic-embed-text
   ```

No other code changes needed — `llm_factory.py` handles the swap transparently.

---

## API Reference

### `POST /api/evaluate`

| Field | Type | Description |
|-------|------|-------------|
| `resume_pdf` | File | PDF file upload |
| `job_description` | Form | Target job description text |

**Response:**
```json
{
  "success": true,
  "final_roast": { ... },
  "retrieved_context": [ ... ],
  "parsed_sections": { ... },
  "error": null
}
```

### `GET /api/health`
Returns Qdrant connectivity status, collection existence, and LLM provider.

### `GET /api/collection-info`
Returns Qdrant collection stats (point count, vector count).

Interactive docs: **http://localhost:8000/docs**

