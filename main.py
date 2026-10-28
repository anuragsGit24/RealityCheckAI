"""
main.py — FastAPI server for "Roast My Resume".

Endpoints
─────────
POST /api/evaluate
    Accepts a resume PDF (multipart) + job description (form field).
    Runs the LangGraph pipeline and returns a structured JSON critique.

GET  /api/health
    Liveness probe — confirms the server and Qdrant are reachable.

GET  /api/collection-info
    Returns stats about the Qdrant collection (useful for demo verification).
"""

from __future__ import annotations

import io
import warnings
from typing import Any

# Filter noisy deprecation warnings on startup
warnings.filterwarnings("ignore", message=".*allowed_objects.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from pypdf import PdfReader
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

from config import get_settings
from graph import roast_graph

# ─────────────────────────────────────────────────────────────────────────────
# App initialisation
# ─────────────────────────────────────────────────────────────────────────────

cfg = get_settings()

app = FastAPI(
    title="Roast My Resume API",
    description=(
        "Evidence-based resume critique powered by Hybrid RAG (Qdrant) + "
        "LangGraph orchestration and LangChain."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Allow all origins for Streamlit ↔ FastAPI local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request, call_next):
    """Log every incoming request and response status."""
    path = request.url.path
    method = request.method
    logger.info(f"➡️  Incoming request: {method} {path}")
    print(f"\n[FASTAPI] ➡️  {method} {path}", flush=True)
    try:
        response = await call_next(request)
        logger.info(f"⬅️  Completed: {method} {path} → {response.status_code}")
        print(f"[FASTAPI] ⬅️  {method} {path} → {response.status_code}", flush=True)
        return response
    except Exception as exc:
        logger.error(f"❌ Error handling {method} {path}: {exc}")
        print(f"[FASTAPI] ❌ Error {method} {path}: {exc}", flush=True)
        raise


# ─────────────────────────────────────────────────────────────────────────────
# Response models
# ─────────────────────────────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    status: str
    qdrant_connected: bool
    qdrant_collection_exists: bool
    llm_provider: str


class EvaluateResponse(BaseModel):
    success: bool
    final_roast: dict[str, Any]
    retrieved_context: list[dict[str, Any]]
    parsed_sections: dict[str, Any]
    error: str | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Extract all text from a PDF byte-stream using pypdf.

    Parameters
    ----------
    file_bytes : Raw bytes of the uploaded PDF.

    Returns
    -------
    Concatenated plain-text content of all pages.
    """
    reader = PdfReader(io.BytesIO(file_bytes))
    text_parts: list[str] = []
    for page in reader.pages:
        t = page.extract_text()
        if t:
            text_parts.append(t.strip())
    return "\n\n".join(text_parts)


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(host=cfg.qdrant_host, port=cfg.qdrant_port, timeout=5)


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────


@app.get("/", tags=["system"])
async def root_status() -> dict[str, Any]:
    """Root endpoint to easily confirm FastAPI is live."""
    return {
        "service": "Roast My Resume API",
        "status": "online",
        "provider": cfg.llm_provider,
        "docs_url": "/docs",
        "endpoints": {
            "evaluate": "POST /api/evaluate",
            "health": "GET /api/health",
            "collection_info": "GET /api/collection-info",
        },
    }


@app.get("/api/health", response_model=HealthResponse, tags=["system"])
async def health_check() -> HealthResponse:
    """
    Liveness probe.

    Checks:
    - FastAPI is running
    - Qdrant is reachable
    - The resume collection exists
    """
    qdrant_ok = False
    collection_ok = False
    try:
        client = get_qdrant_client()
        collections = [c.name for c in client.get_collections().collections]
        qdrant_ok = True
        collection_ok = cfg.qdrant_collection in collections
    except Exception as exc:
        logger.warning(f"Qdrant health check failed: {exc}")

    return HealthResponse(
        status="ok",
        qdrant_connected=qdrant_ok,
        qdrant_collection_exists=collection_ok,
        llm_provider=cfg.llm_provider,
    )


@app.get("/api/collection-info", tags=["system"])
async def collection_info() -> JSONResponse:
    """
    Returns point count and vector config of the Qdrant collection.
    Useful for evaluators to verify the ingestion step.
    """
    try:
        client = get_qdrant_client()
        info = client.get_collection(cfg.qdrant_collection)
        return JSONResponse(
            content={
                "collection": cfg.qdrant_collection,
                "points_count": info.points_count,
                "vectors_count": info.vectors_count,
                "status": str(info.status),
            }
        )
    except UnexpectedResponse as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Collection '{cfg.qdrant_collection}' not found. Run ingest.py first.",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Qdrant unreachable: {exc}",
        ) from exc


@app.post(
    "/api/evaluate",
    response_model=EvaluateResponse,
    tags=["evaluation"],
    summary="Evaluate a resume against a job description",
)
async def evaluate_resume(
    resume_pdf: UploadFile = File(..., description="The candidate's resume as a PDF file."),
    job_description: str = Form(..., description="The target job description text."),
) -> EvaluateResponse:
    """
    Full pipeline endpoint.

    1. Validates the uploaded file is a PDF.
    2. Extracts text from the PDF.
    3. Runs the 3-node LangGraph: parse → retrieve → generate.
    4. Returns the structured roast + retrieved context for evaluator transparency.
    """
    # ── Validate file type ────────────────────────────────────────────────────
    if resume_pdf.content_type not in ("application/pdf", "application/octet-stream"):
        if not (resume_pdf.filename or "").lower().endswith(".pdf"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file must be a PDF.",
            )

    if not job_description or len(job_description.strip()) < 20:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Job description is too short. Please provide at least 20 characters.",
        )

    # ── Extract PDF text ──────────────────────────────────────────────────────
    logger.info(f"Received PDF: '{resume_pdf.filename}'  JD length: {len(job_description)}")
    try:
        pdf_bytes = await resume_pdf.read()
        resume_text = extract_text_from_pdf(pdf_bytes)
    except Exception as exc:
        logger.error(f"PDF extraction failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to parse the PDF: {exc}",
        ) from exc

    if len(resume_text.strip()) < 50:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "The uploaded PDF appears to be empty or image-based (non-text). "
                "Please upload a text-selectable PDF."
            ),
        )

    # ── Run LangGraph pipeline ─────────────────────────────────────────────────
    logger.info("Invoking LangGraph pipeline …")
    initial_state: dict[str, Any] = {
        "resume_text": resume_text,
        "job_description": job_description.strip(),
        "parsed_sections": {},
        "retrieved_context": [],
        "final_roast": {},
        "error": None,
    }

    try:
        final_state: dict[str, Any] = roast_graph.invoke(initial_state)
    except Exception as exc:
        logger.error(f"LangGraph pipeline crashed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline execution failed: {exc}",
        ) from exc

    logger.success("Pipeline completed successfully.")

    return EvaluateResponse(
        success=True,
        final_roast=final_state.get("final_roast", {}),
        retrieved_context=final_state.get("retrieved_context", []),
        parsed_sections=final_state.get("parsed_sections", {}),
        error=final_state.get("error"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Entry-point (for direct execution: python main.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    print("=" * 60, flush=True)
    print("🔥 Starting Roast My Resume FastAPI Backend", flush=True)
    print(f"   Listening on: http://{cfg.api_host}:{cfg.api_port}", flush=True)
    print(f"   Swagger Docs: http://127.0.0.1:{cfg.api_port}/docs", flush=True)
    print(f"   Health Check: http://127.0.0.1:{cfg.api_port}/api/health", flush=True)
    print("=" * 60, flush=True)

    uvicorn.run(
        "main:app",
        host=cfg.api_host,
        port=cfg.api_port,
        reload=False,
        log_level="info",
    )

