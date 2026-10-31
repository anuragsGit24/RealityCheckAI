"""
ingest.py — One-time ingestion script.

Usage
-----
    python ingest.py

What it does
------------
1. Robustly locates and loads the three ground-truth PDFs from ``data/``
   (with smart fuzzy & keyword discovery for filename/dash variations).
2. Splits each document into overlapping chunks via RecursiveCharacterTextSplitter.
3. Assigns fine-grained metadata to every chunk (doc_type, source, page).
4. Generates:
   - Dense embeddings via the configured model (OpenAI / sentence-transformers / Ollama)
   - Sparse BM25 term vectors with deterministic token hashing (Qdrant compatible)
5. Upserts all chunks into local Qdrant with Hybrid Search enabled.
"""

from __future__ import annotations

import math
import os
import re
import sys
import uuid
import zlib
from collections import Counter
from pathlib import Path
from typing import Any

from langchain.text_splitter import RecursiveCharacterTextSplitter
from loguru import logger
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    PointStruct,
    PayloadSchemaType,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
)

from config import get_qdrant_client, get_settings
from llm_factory import get_embeddings

# ── Logging ───────────────────────────────────────────────────────────────────
logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:HH:mm:ss}</green> | {level} | {message}",
)

# ── Constants ─────────────────────────────────────────────────────────────────
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"


# ─────────────────────────────────────────────────────────────────────────────
# Robust File Discovery
# ─────────────────────────────────────────────────────────────────────────────


def normalize_string(s: str) -> str:
    """Normalize string by removing non-alphanumeric chars and lowercasing."""
    return re.sub(r"[^a-zA-Z0-9]", "", s).lower()


def find_pdf_file(data_dir: Path, target_filename: str, doc_type: str) -> Path | None:
    """
    Intelligently find a PDF file matching the target or its semantic category.

    Handles:
    - Unicode em-dash (—) vs regular hyphen (-) vs en-dash (–) vs underscores
    - Files located in `data/`, subfolders of `data/`, or current working dir
    - Semantic keyword matching (e.g. 'applicant' / 'leland' / 'resume')
    """
    # 1. Direct path check in data_dir
    direct = data_dir / target_filename
    if direct.is_file():
        return direct

    # 2. Check root directory
    root_direct = Path(target_filename)
    if root_direct.is_file():
        return root_direct

    # 3. Collect all candidate PDFs
    candidates: list[Path] = []
    if data_dir.exists():
        candidates.extend(list(data_dir.glob("*.pdf")))
        candidates.extend(list(data_dir.glob("**/*.pdf")))
    candidates.extend(list(Path(".").glob("*.pdf")))

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_candidates: list[Path] = []
    for c in candidates:
        resolved = str(c.resolve())
        if resolved not in seen:
            seen.add(resolved)
            unique_candidates.append(c)

    # 4. Normalized exact match (ignores dashes, spaces, case, extensions)
    target_norm = normalize_string(target_filename)
    for c in unique_candidates:
        if normalize_string(c.name) == target_norm:
            logger.info(f"  ✓ Discovered '{c.name}' via normalized match for '{target_filename}'")
            return c

    # 5. Semantic keyword matching based on doc_type
    keyword_map = {
        "ats_rule": ["applicant", "tracking", "ats"],
        "writing_rule": ["leland", "ultimate", "career", "writing"],
        "gold_example": ["resume-book", "resume_book", "mit", "gold", "resume book"],
    }
    keywords = keyword_map.get(doc_type, [])

    for c in unique_candidates:
        c_lower = c.name.lower()
        if any(kw in c_lower for kw in keywords):
            logger.info(f"  ✓ Discovered '{c.name}' for {doc_type} via keyword matching")
            return c

    return None


# ─────────────────────────────────────────────────────────────────────────────
# PDF Loading helpers
# ─────────────────────────────────────────────────────────────────────────────


def load_pdf(path: Path) -> list[dict[str, Any]]:
    """Extract text from every page of a PDF."""
    reader = PdfReader(str(path))
    pages: list[dict[str, Any]] = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            pages.append({"page": i + 1, "text": text})
    logger.info(f"  Loaded {len(pages)} pages from '{path.name}'")
    return pages


def chunk_pages(
    pages: list[dict[str, Any]],
    source: str,
    doc_type: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[dict[str, Any]]:
    """Split pages into overlapping text chunks with attached metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[dict[str, Any]] = []
    for page_data in pages:
        raw_chunks = splitter.split_text(page_data["text"])
        for i, chunk_text in enumerate(raw_chunks):
            chunk_text = chunk_text.strip()
            if len(chunk_text) < 30:  # discard empty/tiny fragments
                continue
            chunks.append(
                {
                    "text": chunk_text,
                    "metadata": {
                        "doc_type": doc_type,
                        "source": source,
                        "page": page_data["page"],
                        "chunk_id": i,
                    },
                }
            )
    logger.info(f"  → {len(chunks)} chunks created  (doc_type={doc_type})")
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Sparse BM25 Token & Vector Computing (Qdrant Compatible)
# ─────────────────────────────────────────────────────────────────────────────


def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase alphanumeric keywords."""
    return re.findall(r"\b[a-zA-Z0-9_\-\.]{2,}\b", text.lower())


def token_to_index(token: str) -> int:
    """
    Deterministically hash a token to a positive 31-bit integer.
    Shared identically across ingestion and real-time query retrieval.
    """
    return (zlib.crc32(token.encode("utf-8")) % 2147483647) + 1


def build_bm25_sparse_vectors(texts: list[str]) -> list[dict[str, Any]]:
    """
    Compute BM25 sparse vectors for a list of document chunks.

    Returns for each text a dict:
        {"indices": [int, ...], "values": [float, ...]}
    formatted directly for Qdrant's sparse vector indexing.
    """
    tokenized_docs = [tokenize(t) for t in texts]
    n_docs = len(tokenized_docs)
    if n_docs == 0:
        return []

    # Compute document frequencies
    doc_freqs: Counter[str] = Counter()
    for doc in tokenized_docs:
        for term in set(doc):
            doc_freqs[term] += 1

    avgdl = sum(len(doc) for doc in tokenized_docs) / max(n_docs, 1)
    k1 = 1.5
    b = 0.75

    sparse_vectors: list[dict[str, Any]] = []
    for doc in tokenized_docs:
        doc_len = len(doc)
        if doc_len == 0:
            sparse_vectors.append({"indices": [], "values": []})
            continue

        tf = Counter(doc)
        term_scores: list[tuple[int, float]] = []

        for term, count in tf.items():
            df = doc_freqs[term]
            # Standard Lucene/BM25 Okapi IDF
            idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            if idf <= 0:
                continue

            # BM25 TF weight
            tf_score = (count * (k1 + 1.0)) / (count + k1 * (1.0 - b + b * (doc_len / avgdl)))
            weight = float(idf * tf_score)
            idx = token_to_index(term)
            term_scores.append((idx, round(weight, 4)))

        # Keep top 256 most significant terms per chunk
        if len(term_scores) > 256:
            term_scores.sort(key=lambda x: x[1], reverse=True)
            term_scores = term_scores[:256]

        # Qdrant requires sorted indices
        term_scores.sort(key=lambda x: x[0])

        indices = [p[0] for p in term_scores]
        values = [p[1] for p in term_scores]
        sparse_vectors.append({"indices": indices, "values": values})

    return sparse_vectors


def build_query_sparse_vector(query_text: str) -> dict[str, list]:
    """
    Compute the sparse vector for a query using identical token index hashing.
    """
    tokens = tokenize(query_text)
    if not tokens:
        return {"indices": [], "values": []}

    tf = Counter(tokens)
    pairs = [(token_to_index(term), float(count)) for term, count in tf.items()]
    pairs.sort(key=lambda x: x[0])

    return {
        "indices": [p[0] for p in pairs],
        "values": [p[1] for p in pairs],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Qdrant setup & upsert
# ─────────────────────────────────────────────────────────────────────────────


def create_qdrant_collection(client: QdrantClient, collection: str, vector_size: int) -> None:
    """Create (or recreate) the Qdrant collection with hybrid-search vectors."""
    existing = [c.name for c in client.get_collections().collections]
    if collection in existing:
        logger.warning(f"Collection '{collection}' exists — recreating for clean state.")
        client.delete_collection(collection)

    client.create_collection(
        collection_name=collection,
        vectors_config={
            DENSE_VECTOR_NAME: VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            )
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: SparseVectorParams(
                index=SparseIndexParams(on_disk=False)
            )
        },
    )
    client.create_payload_index(
        collection_name=collection,
        field_name="doc_type",
        field_schema=PayloadSchemaType.KEYWORD,
    )
    logger.info(f"Collection '{collection}' created (dense={vector_size}d + sparse BM25).")


def upsert_chunks(
    client: QdrantClient,
    collection: str,
    chunks: list[dict[str, Any]],
    dense_embeddings: list[list[float]],
    sparse_vectors: list[dict[str, Any]],
) -> None:
    """Batch-upsert chunk points into Qdrant."""
    points: list[PointStruct] = []
    for chunk, dense, sparse in zip(chunks, dense_embeddings, sparse_vectors):
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector={
                DENSE_VECTOR_NAME: dense,
                SPARSE_VECTOR_NAME: sparse,
            },
            payload={
                "text": chunk["text"],
                **chunk["metadata"],
            },
        )
        points.append(point)

    # Cloud uploads can take longer than local Docker uploads; smaller batches
    # keep each request comfortably within the configured client timeout.
    batch_size = 16
    for start in range(0, len(points), batch_size):
        batch = points[start : start + batch_size]
        client.upsert(collection_name=collection, points=batch)
        logger.info(f"  Upserted points {start}–{start + len(batch) - 1}")


# ─────────────────────────────────────────────────────────────────────────────
# Main entry-point
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    cfg = get_settings()
    data_dir = Path(cfg.data_dir)

    manifest = [
        {
            "filename": cfg.ats_pdf,
            "doc_type": "ats_rule",
            "label": "ATS Guide",
        },
        {
            "filename": cfg.leland_pdf,
            "doc_type": "writing_rule",
            "label": "Leland Writing Guide",
        },
        {
            "filename": cfg.mit_pdf,
            "doc_type": "gold_example",
            "label": "MIT Resume Book",
        },
    ]

    all_chunks: list[dict[str, Any]] = []

    logger.info("=" * 60)
    logger.info("PHASE 1 — Discovering, Loading & Chunking PDFs")
    logger.info("=" * 60)

    for doc in manifest:
        pdf_path = find_pdf_file(data_dir, doc["filename"], doc["doc_type"])
        if not pdf_path:
            logger.error(f"❌ PDF not found for {doc['label']} ('{doc['filename']}').")
            continue

        logger.info(f"\n📄  {doc['label']} ({doc['doc_type']}) → {pdf_path}")
        pages = load_pdf(pdf_path)
        chunks = chunk_pages(
            pages,
            source=pdf_path.name,
            doc_type=doc["doc_type"],
            chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap,
        )
        all_chunks.extend(chunks)

    if not all_chunks:
        # Diagnostic directory listing
        existing_pdfs = list(data_dir.glob("*.pdf")) if data_dir.exists() else []
        root_pdfs = list(Path(".").glob("*.pdf"))
        logger.error(
            f"No chunks produced! Check your data directory ('{data_dir}').\n"
            f"PDFs found in '{data_dir}': {[p.name for p in existing_pdfs]}\n"
            f"PDFs found in root: {[p.name for p in root_pdfs]}"
        )
        sys.exit(1)

    logger.info(f"\nTotal chunks across all loaded documents: {len(all_chunks)}")

    # ── Embeddings ────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 2 — Generating Dense Embeddings")
    logger.info("=" * 60)

    embed_model = get_embeddings()
    texts = [c["text"] for c in all_chunks]

    logger.info(f"Embedding {len(texts)} chunks …")
    dense_embeddings = embed_model.embed_documents(texts)
    vector_size = len(dense_embeddings[0])
    logger.info(f"Dense embedding dimension: {vector_size}")

    # ── Sparse BM25 vectors ───────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 3 — Computing Sparse BM25 Vectors")
    logger.info("=" * 60)

    sparse_vecs = build_bm25_sparse_vectors(texts)
    logger.info(f"BM25 sparse vectors computed for {len(sparse_vecs)} chunks.")

    # ── Qdrant ────────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 4 — Upserting into Qdrant")
    logger.info("=" * 60)

    # Local Docker Qdrant:
    # client = QdrantClient(host=cfg.qdrant_host, port=cfg.qdrant_port)
    client = get_qdrant_client(cfg)
    create_qdrant_collection(client, cfg.qdrant_collection, vector_size)
    upsert_chunks(client, cfg.qdrant_collection, all_chunks, dense_embeddings, sparse_vecs)

    # ── Summary ───────────────────────────────────────────────────────────────
    doc_type_counts: dict[str, int] = {}
    for c in all_chunks:
        dt = c["metadata"]["doc_type"]
        doc_type_counts[dt] = doc_type_counts.get(dt, 0) + 1

    logger.info("\n" + "=" * 60)
    logger.info("✅  INGESTION COMPLETE")
    logger.info("=" * 60)
    for dt, cnt in doc_type_counts.items():
        logger.info(f"  {dt:20s}: {cnt} chunks")
    logger.info(f"  {'TOTAL':20s}: {len(all_chunks)} chunks")
    logger.info(f"\nQdrant collection '{cfg.qdrant_collection}' is ready for search.")


if __name__ == "__main__":
    main()
