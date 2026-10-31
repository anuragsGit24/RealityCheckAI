"""
graph.py — LangGraph State Machine for "Roast My Resume".

The pipeline is a strictly sequential, 3-node LangGraph:

    ┌────────────┐    ┌──────────────┐    ┌────────────────┐
    │ node_parse │ → │ node_retrieve │ → │ node_generate  │
    └────────────┘    └──────────────┘    └────────────────┘

State schema  (GraphState TypedDict)
────────────────────────────────────
resume_text      : str          Raw extracted resume text from PDF
job_description  : str          Target JD provided by the user
parsed_sections  : dict         Structured JSON: Experience, Education, Skills
retrieved_context: list[dict]   Chunks returned by Qdrant hybrid search
final_roast      : dict         "brutal_roast" + "actionable_fixes" from LLM
error            : str | None   Set on any unrecoverable node failure
"""

from __future__ import annotations

import json
import re
from typing import Any, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph
from loguru import logger
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient, models

from config import get_qdrant_client, get_settings
from llm_factory import get_embeddings, get_llm
from ingest import build_bm25_sparse_vectors, build_query_sparse_vector

# ─────────────────────────────────────────────────────────────────────────────
# Graph State
# ─────────────────────────────────────────────────────────────────────────────


class GraphState(TypedDict, total=False):
    """
    Shared mutable state passed between LangGraph nodes.

    All fields are optional (total=False) so nodes only need to return
    the keys they update — LangGraph merges the dict automatically.
    """

    resume_text: str
    job_description: str
    parsed_sections: dict[str, Any]
    retrieved_context: list[dict[str, Any]]
    final_roast: dict[str, Any]
    retrieval_diagnostics: dict[str, Any]
    error: str | None


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic output schemas (for structured LLM output)
# ─────────────────────────────────────────────────────────────────────────────


class ParsedResume(BaseModel):
    """Structured resume sections extracted by node_parse."""

    work_experience: list[str] = Field(
        description="All bullet points or sentences from Work Experience / Projects."
    )
    education: list[str] = Field(
        description="All education entries (degree, institution, dates, GPA)."
    )
    skills: list[str] = Field(
        description="All listed technical skills, tools, and languages."
    )
    summary: str = Field(
        default="",
        description="Professional summary or objective statement, if present.",
    )


class ActionableFix(BaseModel):
    """A single, evidence-backed improvement suggestion."""

    issue: str = Field(description="The specific problem found in the resume.")
    fix: str = Field(description="Concrete, actionable rewrite or improvement.")
    rule_reference: str = Field(
        description="The exact retrieved rule or gold-example snippet that justifies this fix."
    )
    severity: str = Field(
        description="Impact level: 'critical', 'major', or 'minor'.",
        default="major",
    )


class RoastOutput(BaseModel):
    """Final structured output from node_generate."""

    overall_score: int = Field(
        description="Resume score out of 100, based on ATS + writing quality + JD alignment.",
        ge=0,
        le=100,
    )
    brutal_roast: str = Field(
        description=(
            "A blunt, 3–5 sentence narrative paragraph that ruthlessly critiques the resume "
            "against the job description. Must reference specific resume weaknesses."
        )
    )
    actionable_fixes: list[ActionableFix] = Field(
        description="Ordered list of actionable fixes (most critical first).",
        min_length=3,
    )
    ats_verdict: str = Field(
        description="'PASS' or 'FAIL' — whether the resume would likely clear ATS screening."
    )
    jd_alignment_score: int = Field(
        description="Keyword / skills alignment score between resume and JD (0-100).",
        ge=0,
        le=100,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Node 1 — Parse Resume
# ─────────────────────────────────────────────────────────────────────────────

_PARSE_SYSTEM = """You are a precise resume parser. Your task is to extract structured information
from the raw resume text provided by the user.

Return ONLY valid JSON matching this exact schema:
{
  "work_experience": ["<bullet or sentence>", ...],
  "education": ["<education entry>", ...],
  "skills": ["<skill>", ...],
  "summary": "<summary text or empty string>"
}

Rules:
- Include EVERY bullet point from Work Experience and Projects verbatim.
- Do NOT paraphrase or omit content.
- If a section is missing, return an empty list [] for that key.
- Output ONLY the JSON object — no markdown fences, no commentary.
"""

_PARSE_HUMAN = """Extract the resume sections from the following raw resume text:

--- RESUME TEXT START ---
{resume_text}
--- RESUME TEXT END ---"""


def node_parse(state: GraphState) -> GraphState:
    """
    Node 1 — Resume Parser.

    Takes raw `resume_text` from state and uses the LLM to extract
    Work Experience, Education, and Skills into structured JSON.
    Updates state with `parsed_sections`.
    """
    logger.info("▶  node_parse  — Extracting resume sections …")

    resume_text = state.get("resume_text", "").strip()
    if not resume_text:
        logger.error("node_parse: resume_text is empty.")
        return {**state, "error": "Resume text is empty.", "parsed_sections": {}}

    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=_PARSE_SYSTEM),
            HumanMessage(content=_PARSE_HUMAN.format(resume_text=resume_text[:6000])),
        ]
    )

    try:
        response = llm.invoke(prompt.format_messages())
        raw = response.content.strip()

        # Strip markdown fences if the model added them
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)

        parsed = json.loads(raw)
        # Validate / fill missing keys
        for key in ("work_experience", "education", "skills"):
            parsed.setdefault(key, [])
        parsed.setdefault("summary", "")

        logger.success(
            f"  Parsed: {len(parsed['work_experience'])} experience bullets, "
            f"{len(parsed['education'])} education entries, "
            f"{len(parsed['skills'])} skills."
        )
        return {**state, "parsed_sections": parsed, "error": None}

    except (json.JSONDecodeError, Exception) as exc:
        logger.error(f"node_parse failed: {exc}")
        # Graceful fallback — return raw text in experience bucket
        fallback = {
            "work_experience": [resume_text[:3000]],
            "education": [],
            "skills": [],
            "summary": "",
        }
        return {**state, "parsed_sections": fallback, "error": str(exc)}


# ─────────────────────────────────────────────────────────────────────────────
# Node 2 — Hybrid Retriever
# ─────────────────────────────────────────────────────────────────────────────

_DENSE_VECTOR_NAME  = "dense"
_SPARSE_VECTOR_NAME = "sparse"


def _hybrid_search(
    client: QdrantClient,
    collection: str,
    query_text: str,
    doc_type_filter: str | None,
    top_k: int,
    embed_model: Any,
) -> list[dict[str, Any]]:
    """
    Run a hybrid search (dense cosine + sparse BM25 re-rank) on Qdrant.

    When *doc_type_filter* is set, only chunks with that metadata tag are
    searched, keeping results semantically relevant.
    """
    cfg = get_settings()

    # 1. Dense embedding for the query
    dense_vec: list[float] = embed_model.embed_query(query_text)

    # 2. Sparse BM25 vector for the query
    sparse_vec = build_query_sparse_vector(query_text)

    # 3. Build filter (if requested)
    qdrant_filter = None
    if doc_type_filter:
        qdrant_filter = models.Filter(
            must=[models.FieldCondition(key="doc_type", match=models.MatchValue(value=doc_type_filter))]
        )

    results: list[dict[str, Any]] = []
    search_errors: list[str] = []

    # 4. Dense-vector search
    try:
        dense_hits = client.query_points(
            collection_name=collection,
            query=dense_vec,
            using=_DENSE_VECTOR_NAME,
            query_filter=qdrant_filter,
            limit=top_k,
            with_payload=True,
        )
        for hit in dense_hits.points:
            payload = hit.payload or {}
            results.append(
                {
                    "text": payload.get("text", ""),
                    "doc_type": payload.get("doc_type", ""),
                    "source": payload.get("source", ""),
                    "page": payload.get("page", 0),
                    "score": round(hit.score, 4),
                    "retrieval_method": "dense",
                }
            )
    except Exception as exc:
        search_errors.append(f"dense: {exc}")
        logger.warning(f"Dense search failed: {exc}")

    # 5. Sparse BM25 search (keyword-level)
    try:
        if sparse_vec and sparse_vec.get("indices"):
            sparse_hits = client.query_points(
                collection_name=collection,
                query=models.SparseVector(
                    indices=sparse_vec["indices"],
                    values=sparse_vec["values"],
                ),
                using=_SPARSE_VECTOR_NAME,
                query_filter=qdrant_filter,
                limit=top_k,
                with_payload=True,
            )
            existing_texts = {r["text"] for r in results}
            for hit in sparse_hits.points:
                payload = hit.payload or {}
                t = payload.get("text", "")
                if t not in existing_texts:
                    results.append(
                        {
                            "text": t,
                            "doc_type": payload.get("doc_type", ""),
                            "source": payload.get("source", ""),
                            "page": payload.get("page", 0),
                            "score": round(hit.score, 4),
                            "retrieval_method": "sparse_bm25",
                        }
                    )
    except Exception as exc:
        search_errors.append(f"sparse: {exc}")
        logger.warning(f"Sparse search failed (non-fatal): {exc}")

    if search_errors and not results:
        raise RuntimeError(
            f"No {doc_type_filter or 'unfiltered'} Qdrant results. "
            + " | ".join(search_errors)
        )

    # Sort merged results by score descending, keep top_k
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


def node_retrieve(state: GraphState) -> GraphState:
    """
    Node 2 — Hybrid RAG Retriever.

    Constructs three targeted queries:
      1. ATS formatting rules (doc_type='ats_rule')
      2. Writing / bullet improvement rules (doc_type='writing_rule')
      3. Gold-standard MIT examples (doc_type='gold_example')

    Merges all results into `retrieved_context`.
    """
    logger.info("▶  node_retrieve  — Querying Qdrant (Hybrid Search) …")

    cfg = get_settings()
    parsed  = state.get("parsed_sections", {})
    jd_text = state.get("job_description", "")
    resume  = state.get("resume_text", "")

    # Local Docker Qdrant:
    # client = QdrantClient(host=cfg.qdrant_host, port=cfg.qdrant_port)
    client      = get_qdrant_client(cfg)
    embed_model = get_embeddings()
    collection  = cfg.qdrant_collection
    k           = cfg.top_k

    # Fail before issuing searches when the configured collection has not been
    # ingested.  The individual dense/sparse searches intentionally tolerate
    # one vector type failing, so without this check a missing collection was
    # silently converted into an empty evidence list.
    collections = {item.name for item in client.get_collections().collections}
    if collection not in collections:
        available = ", ".join(sorted(collections)) or "none"
        raise RuntimeError(
            f"Qdrant collection '{collection}' does not exist at "
            f"{cfg.qdrant_url} (available: {available}). "
            "Run 'python ingest.py' to create and populate it."
        )

    collection_info = client.get_collection(collection)
    point_count = collection_info.points_count or 0
    if not point_count:
        raise RuntimeError(
            f"Qdrant collection '{collection}' exists but contains no points. "
            "Run 'python ingest.py' to populate it."
        )

    # Metadata-filtered searches require a Qdrant payload index. Create it
    # here as well as during ingestion so existing populated collections work
    # without requiring a destructive re-ingestion.
    client.create_payload_index(
        collection_name=collection,
        field_name="doc_type",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )

    retrieval_diagnostics: dict[str, Any] = {
        "backend": "Qdrant Cloud",
        "collection": collection,
        "points": point_count,
        "url": cfg.qdrant_url,
        "queries": [],
    }

    experience_text = " ".join(parsed.get("work_experience", [])[:5])
    skills_text     = " ".join(parsed.get("skills", [])[:20])

    # Query 1 — ATS formatting rules
    ats_query = (
        f"ATS machine readability formatting rules for a resume applying to: {jd_text[:300]}"
    )
    try:
        ats_results = _hybrid_search(client, collection, ats_query, "ats_rule", k, embed_model)
        ats_error = None
    except Exception as exc:
        ats_results = []
        ats_error = str(exc)
    logger.info(f"  ATS rules retrieved: {len(ats_results)}")
    retrieval_diagnostics["queries"].append(
        {"doc_type": "ats_rule", "results": len(ats_results), "error": ats_error}
    )

    # Query 2 — Writing formulas
    writing_query = (
        f"Improve weak resume bullet points using XYZ formula impact metrics action verbs. "
        f"Current bullets: {experience_text[:300]}"
    )
    try:
        writing_results = _hybrid_search(client, collection, writing_query, "writing_rule", k, embed_model)
        writing_error = None
    except Exception as exc:
        writing_results = []
        writing_error = str(exc)
    logger.info(f"  Writing rules retrieved: {len(writing_results)}")
    retrieval_diagnostics["queries"].append(
        {"doc_type": "writing_rule", "results": len(writing_results), "error": writing_error}
    )

    # Query 3 — Gold MIT examples similar to the user's experience
    example_query = (
        f"Gold standard engineering resume example for skills: {skills_text[:200]} "
        f"applying for: {jd_text[:200]}"
    )
    try:
        example_results = _hybrid_search(client, collection, example_query, "gold_example", k, embed_model)
        example_error = None
    except Exception as exc:
        example_results = []
        example_error = str(exc)
    logger.info(f"  Gold examples retrieved: {len(example_results)}")
    retrieval_diagnostics["queries"].append(
        {"doc_type": "gold_example", "results": len(example_results), "error": example_error}
    )

    all_context = ats_results + writing_results + example_results
    logger.success(f"  Total retrieved context chunks: {len(all_context)}")

    return {
        **state,
        "retrieved_context": all_context,
        "retrieval_diagnostics": retrieval_diagnostics,
        "error": None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node 3 — Generate Roast
# ─────────────────────────────────────────────────────────────────────────────

_ROAST_SYSTEM = """You are the senior resume evaluator at a brutally selective technical recruiting firm.
You have reviewed thousands of engineering, product, data, and software resumes. Your voice is sharp,
observant, expressive, and occasionally witty, but never childish, abusive, or personal. Roast the
document and its decisions—not the candidate's worth.

Your job is to expose the difference between what this resume claims and what a skeptical hiring manager
can actually verify in a six-second scan. Treat every sentence as evidence: identify what is vague,
unmeasured, repetitive, inflated, buried, poorly ordered, or unsupported. Do not reward fashionable
phrases merely because they sound professional.

Evaluation standard:
- ATS READABILITY: Check structure, section clarity, conventional headings, parsing risks, keyword
  coverage, and whether important evidence is likely to survive automated screening.
- IMPACT: Check whether bullets show ownership, action, scope, technical difficulty, measurable results,
  and clear before/after value. Separate responsibilities from achievements.
- SIGNAL-TO-NOISE: Penalize filler, generic adjectives, duplicated claims, unexplained acronyms,
  bloated summaries, skill dumps, and details that consume space without improving candidacy.
- JD FIT: Compare the resume to the supplied job description. Identify missing requirements, weak
  keyword alignment, and claims that are present but not demonstrated.
- CREDIBILITY: Never invent metrics, employers, technologies, outcomes, or experience. If a number is
  missing, propose a measurable placeholder or a truthful way to quantify it rather than fabricating one.
- GROUNDING: Use the retrieved context as the benchmark for recommendations. If the context does not
  support a claim, label the recommendation as general resume judgment and do not manufacture a quote.

Writing requirements:
- Start the narrative with the most consequential diagnosis, not a generic compliment.
- Make the narrative vivid and memorable through precise comparisons and strong verbs, while remaining
  professionally useful.
- Reference the candidate's actual wording, section, role, date, or claim whenever possible.
- Distinguish fatal or high-risk issues from polish. Do not treat every imperfection as critical.
- Every fix must say exactly what to change and provide a ready-to-adapt rewrite or transformation.
- Preserve truthful content. Do not add decorative symbols to the resume or treat ordinary bullet markers
  as decorative symbols; only discuss a symbol if it is literally present in the supplied resume text.
- Order fixes from highest hiring impact to lowest. Include at least three fixes when the resume has
  enough evidence, and do not pad the list with repetitive advice.

Return ONLY valid JSON matching this exact schema (no markdown fences and no extra keys):
{
  "overall_score": <int 0-100>,
  "brutal_roast": "<4-7 sentence sharp, expressive, evidence-backed narrative>",
  "actionable_fixes": [
    {
      "issue": "<specific problem>",
      "fix": "<concrete rewrite, replacement structure, or action>",
      "rule_reference": "<short exact quote from retrieved context, or 'General resume judgment: <specific principle>'>",
      "severity": "critical|major|minor"
    }
  ],
  "ats_verdict": "PASS|FAIL",
  "jd_alignment_score": <int 0-100>
}
"""

_ROAST_HUMAN = """
=== JOB DESCRIPTION ===
{job_description}

=== CANDIDATE RESUME (Raw Text) ===
{resume_text}

=== RETRIEVED GROUND-TRUTH CONTEXT FROM VECTOR DATABASE ===
{context_block}

Now generate the evidence-based roast JSON:
"""


def _format_context_block(context: list[dict[str, Any]]) -> str:
    """Format retrieved chunks into a numbered block for the LLM prompt."""
    lines = []
    for i, chunk in enumerate(context, 1):
        lines.append(
            f"[{i}] ({chunk['doc_type'].upper()} | {chunk['source']} | p.{chunk['page']} | "
            f"score={chunk['score']})\n{chunk['text']}"
        )
    return "\n\n".join(lines)


def node_generate(state: GraphState) -> GraphState:
    """
    Node 3 — Roast Generator.

    Combines resume, JD, and Qdrant context in a structured LLM prompt
    and produces the final `final_roast` dict.
    """
    logger.info("▶  node_generate  — Synthesising evidence-based roast …")

    resume_text = state.get("resume_text", "")
    jd          = state.get("job_description", "")
    context     = state.get("retrieved_context", [])

    context_block = _format_context_block(context)

    llm = get_llm()

    messages = [
        SystemMessage(content=_ROAST_SYSTEM),
        HumanMessage(
            content=_ROAST_HUMAN.format(
                job_description=jd[:2000],
                resume_text=resume_text[:4000],
                context_block=context_block[:5000],
            )
        ),
    ]

    try:
        response = llm.invoke(messages)
        raw = response.content.strip()

        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)

        roast_dict = json.loads(raw)

        # Validate required keys
        for key in ("overall_score", "brutal_roast", "actionable_fixes", "ats_verdict"):
            if key not in roast_dict:
                raise ValueError(f"Missing required key in LLM response: {key}")

        roast_dict.setdefault("jd_alignment_score", 0)

        logger.success(
            f"  Roast generated — Score: {roast_dict['overall_score']}/100  "
            f"ATS: {roast_dict['ats_verdict']}  "
            f"Fixes: {len(roast_dict['actionable_fixes'])}"
        )
        return {**state, "final_roast": roast_dict, "error": None}

    except (json.JSONDecodeError, ValueError, Exception) as exc:
        logger.error(f"node_generate failed: {exc}")
        fallback_roast = {
            "overall_score": 0,
            "brutal_roast": f"The roast generation failed due to a system error: {exc}",
            "actionable_fixes": [],
            "ats_verdict": "UNKNOWN",
            "jd_alignment_score": 0,
        }
        return {**state, "final_roast": fallback_roast, "error": str(exc)}


# ─────────────────────────────────────────────────────────────────────────────
# Graph Assembly
# ─────────────────────────────────────────────────────────────────────────────


def build_graph() -> Any:
    """
    Compile and return the LangGraph StateGraph.

    The graph is strictly sequential (no cycles) for demo stability:
        START → node_parse → node_retrieve → node_generate → END
    """
    workflow = StateGraph(GraphState)

    workflow.add_node("node_parse",    node_parse)
    workflow.add_node("node_retrieve", node_retrieve)
    workflow.add_node("node_generate", node_generate)

    workflow.set_entry_point("node_parse")
    workflow.add_edge("node_parse",    "node_retrieve")
    workflow.add_edge("node_retrieve", "node_generate")
    workflow.add_edge("node_generate", END)

    return workflow.compile()


# Singleton compiled graph (imported by main.py)
roast_graph = build_graph()
logger.info("LangGraph compiled successfully — 3-node sequential pipeline ready.")
