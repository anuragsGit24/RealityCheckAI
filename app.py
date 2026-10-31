"""
app.py — RealityCheck AI: Enterprise Peer-Reviewed Career Demolition Interface

A modern, high-precision SaaS frontend for evaluating resumes against empirical
ground-truth documents (ATS guidelines, Leland bullet formulas, and MIT benchmarks)
using Hybrid RAG and deterministic LangGraph state machine orchestration.

Architecture:
- Front-and-center input architecture (Candidate Dossier + Target Specification).
- UX State Management with progressive multi-step `st.status()` telemetry.
- Tabbed Results Dashboard:
    - Tab 1: Empirical Roast (Executive Pathologist Report)
    - Tab 2: Prescriptive Remediation Protocol (Actionable Fixes with Citations)
    - Tab 3: RAG Retrieval Evidence Base (Evaluator Inspection Log)
    - Tab 4: Syntactic Schema (Extracted Candidate AST)
- Dual-Mode Reliability: High-speed FastAPI REST client with seamless in-process
  LangGraph engine fallback for 100% demo uptime.
"""

from __future__ import annotations

import html
import json
import time
from typing import Any

import httpx
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Backend Settings & Telemetry Discovery
# ─────────────────────────────────────────────────────────────────────────────

try:
    from config import get_settings as _get_settings
    from llm_factory import provider_info as _provider_info
    _LLM_INFO = _provider_info()
    _LLM_CFG = _get_settings()
except Exception:
    _LLM_INFO = {}
    _LLM_CFG = None

API_PORT = getattr(_LLM_CFG, "api_port", 8000) if _LLM_CFG else 8000
CANDIDATE_BASES = [
    f"http://127.0.0.1:{API_PORT}",
    f"http://localhost:{API_PORT}",
]

# ─────────────────────────────────────────────────────────────────────────────
# Streamlit Page Configuration
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="RealityCheck AI",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# Global SaaS Stylesheet (Clean Dark Mode, Typography, Satirical Precision)
# ─────────────────────────────────────────────────────────────────────────────

GLOBAL_THEME_CSS = """
<style>
/* ── Typography & Global Reset ── */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&family=Newsreader:ital,opsz,wght@1,6..72,400;1,6..72,500;1,6..72,600&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #e2e8f0;
}

/* ── Hide Default Streamlit Clutter ── */
#MainMenu { visibility: hidden !important; }
header[data-testid="stHeader"] { visibility: hidden !important; height: 0 !important; }
footer { visibility: hidden !important; }
[data-testid="stToolbar"] { visibility: hidden !important; }
[data-testid="stDecoration"] { display: none !important; }

/* ── App Canvas ── */
.stApp {
    background: radial-gradient(circle at 50% 0%, #171d2d 0%, #0c1017 55%, #07090e 100%);
    background-attachment: fixed;
}

.block-container {
    padding-top: 2rem !important;
    padding-bottom: 4rem !important;
    max-width: 1200px !important;
}

/* ── Hero & Badges ── */
.hero-wrapper {
    text-align: center;
    padding: 1.5rem 0 2rem 0;
    margin-bottom: 1.5rem;
    position: relative;
}

.meta-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(244, 63, 94, 0.1);
    border: 1px solid rgba(244, 63, 94, 0.25);
    color: #fda4af;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 5px 14px;
    border-radius: 9999px;
    margin-bottom: 1rem;
}

.hero-title {
    font-size: 3rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    line-height: 1.15;
    margin-bottom: 0.75rem;
    background: linear-gradient(135deg, #ffffff 30%, #cbd5e1 70%, #94a3b8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-title span.accent {
    background: linear-gradient(135deg, #f43f5e 0%, #fb7185 50%, #fda4af 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    font-size: 1.05rem;
    color: #94a3b8;
    max-width: 740px;
    margin: 0 auto;
    line-height: 1.6;
    font-weight: 400;
}

/* ── Panel Cards & Containers ── */
.saas-card {
    background: rgba(18, 24, 38, 0.75);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 1.4rem;
    backdrop-filter: blur(12px);
    transition: border-color 0.2s ease, transform 0.2s ease;
    height: 100%;
}

.saas-card:hover {
    border-color: rgba(255, 255, 255, 0.16);
}

.saas-card-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1rem;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
}

.saas-card-title {
    font-size: 0.95rem;
    font-weight: 700;
    color: #f1f5f9;
    letter-spacing: -0.01em;
    display: flex;
    align-items: center;
    gap: 8px;
}

.saas-card-caption {
    font-size: 0.75rem;
    color: #64748b;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Metrics Dashboard Grid ── */
.metric-box {
    background: rgba(15, 22, 36, 0.85);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 12px;
    padding: 1.1rem 1rem;
    text-align: center;
}

.metric-box .metric-label {
    font-size: 0.72rem;
    font-weight: 700;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.35rem;
}

.metric-box .metric-value {
    font-size: 2.2rem;
    font-weight: 800;
    line-height: 1;
    letter-spacing: -0.02em;
}

.metric-box .metric-foot {
    font-size: 0.7rem;
    color: #64748b;
    margin-top: 0.4rem;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Brutal Roast Narrative Card ── */
.roast-manifesto {
    background: linear-gradient(180deg, rgba(225, 29, 72, 0.08) 0%, rgba(15, 23, 42, 0.6) 100%);
    border: 1px solid rgba(244, 63, 94, 0.3);
    border-radius: 14px;
    padding: 1.8rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}

.roast-manifesto::before {
    content: "CRITIQUE";
    position: absolute;
    top: 12px;
    right: 20px;
    font-size: 3.5rem;
    font-weight: 900;
    color: rgba(244, 63, 94, 0.04);
    pointer-events: none;
    letter-spacing: 0.05em;
    font-family: 'Plus Jakarta Sans', sans-serif;
}

.roast-quote-badge {
    display: inline-block;
    padding: 3px 10px;
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    border-radius: 6px;
    background: rgba(244, 63, 94, 0.2);
    color: #f43f5e;
    margin-bottom: 1rem;
}

.roast-body {
    font-family: 'Newsreader', Georgia, serif;
    font-size: 1.25rem;
    line-height: 1.7;
    font-style: italic;
    color: #f8fafc;
}

/* ── Actionable Fix Row ── */
.fix-entry {
    background: rgba(15, 23, 42, 0.65);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 10px;
    padding: 1.1rem 1.25rem;
    margin-bottom: 0.85rem;
    border-left-width: 4px;
    transition: transform 0.15s ease, border-color 0.15s ease;
}

.fix-entry:hover {
    border-color: rgba(255, 255, 255, 0.14);
}

.fix-entry.critical { border-left-color: #f43f5e; }
.fix-entry.major    { border-left-color: #f59e0b; }
.fix-entry.minor    { border-left-color: #10b981; }

.fix-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 0.5rem;
    flex-wrap: wrap;
}

.fix-index {
    font-size: 0.72rem;
    font-family: 'JetBrains Mono', monospace;
    color: #64748b;
    font-weight: 600;
}

.fix-issue-title {
    font-size: 0.95rem;
    font-weight: 700;
    color: #f1f5f9;
}

.sev-badge {
    margin-left: auto;
    font-size: 0.65rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    padding: 2px 8px;
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace;
}

.sev-badge.critical { background: rgba(244, 63, 94, 0.15); color: #fb7185; }
.sev-badge.major    { background: rgba(245, 158, 11, 0.15); color: #fbbf24; }
.sev-badge.minor    { background: rgba(16, 185, 129, 0.15); color: #34d399; }

.fix-rewrite {
    font-size: 0.9rem;
    color: #cbd5e1;
    margin-bottom: 0.6rem;
    line-height: 1.5;
}

.fix-citation {
    font-size: 0.78rem;
    color: #94a3b8;
    background: rgba(0, 0, 0, 0.25);
    border-radius: 6px;
    padding: 6px 10px;
    border-left: 2px solid #3b82f6;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Context & Ground Truth Card ── */
.rag-chunk {
    background: rgba(11, 17, 30, 0.8);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 8px;
    padding: 0.9rem;
    margin-bottom: 0.7rem;
    font-size: 0.82rem;
    line-height: 1.55;
    font-family: 'JetBrains Mono', monospace;
}

.rag-chunk-meta {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 0.4rem;
    font-size: 0.72rem;
    color: #64748b;
    flex-wrap: wrap;
}

.rag-tag {
    font-weight: 700;
    padding: 2px 6px;
    border-radius: 4px;
    text-transform: uppercase;
}

.rag-tag.ats_rule     { background: #1e3a5f; color: #93c5fd; }
.rag-tag.writing_rule { background: #3b2a1a; color: #fcd34d; }
.rag-tag.gold_example { background: #183d2b; color: #6ee7b7; }

/* ── Custom Primary Button ── */
div.stButton > button {
    background: linear-gradient(135deg, #e11d48 0%, #be123c 60%, #9f1239 100%) !important;
    color: #ffffff !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 10px !important;
    padding: 0.75rem 2rem !important;
    font-weight: 700 !important;
    font-size: 0.98rem !important;
    letter-spacing: -0.01em !important;
    box-shadow: 0 4px 18px rgba(225, 29, 72, 0.35) !important;
    transition: all 0.22s ease-in-out !important;
    width: 100% !important;
}

div.stButton > button:hover {
    background: linear-gradient(135deg, #f43f5e 0%, #e11d48 60%, #be123c 100%) !important;
    box-shadow: 0 6px 22px rgba(244, 63, 94, 0.45) !important;
    transform: translateY(-1px) !important;
    color: #ffffff !important;
}

div.stButton > button:disabled {
    background: rgba(30, 41, 59, 0.6) !important;
    color: #64748b !important;
    border-color: rgba(255, 255, 255, 0.05) !important;
    box-shadow: none !important;
    transform: none !important;
}

/* ── Clean Tab Bars ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: transparent;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    padding-bottom: 4px;
}

.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border: none !important;
    color: #94a3b8 !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    padding: 8px 16px !important;
    border-radius: 6px !important;
}

.stTabs [aria-selected="true"] {
    color: #ffffff !important;
    background: rgba(255, 255, 255, 0.06) !important;
    border-bottom: 2px solid #f43f5e !important;
}

/* ── Top Bar Telemetry Badge ── */
.top-telemetry-bar {
    display: flex;
    justify-content: flex-end;
    align-items: center;
    gap: 12px;
    margin-bottom: 1rem;
    font-size: 0.75rem;
    color: #64748b;
    font-family: 'JetBrains Mono', monospace;
}
.telemetry-indicator {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(15, 23, 42, 0.6);
    padding: 3px 10px;
    border-radius: 9999px;
    border: 1px solid rgba(255, 255, 255, 0.06);
}
.dot-live {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background-color: #10b981;
    box-shadow: 0 0 8px #10b981;
}
</style>
"""

st.markdown(GLOBAL_THEME_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Backend API & Direct Engine Gateway
# ─────────────────────────────────────────────────────────────────────────────


def check_system_telemetry() -> dict[str, Any] | None:
    """Probes candidate backend URLs for system health and provider details."""
    for base in CANDIDATE_BASES:
        try:
            r = httpx.get(f"{base}/api/health", timeout=2.5)
            if r.status_code == 200:
                data = r.json()
                data["active_base"] = base
                return data
        except Exception:
            pass
    return None


def execute_evaluation(pdf_bytes: bytes, job_description: str) -> dict[str, Any]:
    """
    Submits evaluation payload to FastAPI REST API with automatic, seamless
    in-process LangGraph engine fallback if the server is offline.
    """
    # 1. Attempt FastAPI HTTP REST endpoint
    for base in CANDIDATE_BASES:
        endpoint = f"{base}/api/evaluate"
        try:
            with httpx.Client(timeout=300) as client:
                response = client.post(
                    endpoint,
                    files={"resume_pdf": ("candidate_resume.pdf", pdf_bytes, "application/pdf")},
                    data={"job_description": job_description},
                )
            if response.status_code == 200:
                payload = response.json()
                payload["execution_runtime"] = f"FastAPI Worker ({base})"
                return payload
        except Exception:
            pass

    # 2. In-Process LangGraph Fallback (Ensures 100% demo continuity)
    from main import extract_text_from_pdf
    from graph import roast_graph

    resume_text = extract_text_from_pdf(pdf_bytes)
    initial_state = {
        "resume_text": resume_text,
        "job_description": job_description.strip(),
        "parsed_sections": {},
        "retrieved_context": [],
        "final_roast": {},
        "error": None,
    }
    final_state = roast_graph.invoke(initial_state)

    return {
        "success": True,
        "execution_runtime": "Direct In-Process LangGraph Engine",
        "final_roast": final_state.get("final_roast", {}),
        "retrieved_context": final_state.get("retrieved_context", []),
        "parsed_sections": final_state.get("parsed_sections", {}),
        "retrieval_diagnostics": final_state.get("retrieval_diagnostics", {}),
        "error": final_state.get("error"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# UI Presentation Components
# ─────────────────────────────────────────────────────────────────────────────


def render_telemetry_bar() -> None:
    """Renders top-right live engine and model diagnostics."""
    telemetry = check_system_telemetry()
    provider_name = _LLM_INFO.get("provider", "Groq / Local")
    model_name = _LLM_INFO.get("chat_model", "llama-3.1-8b-instant")

    if telemetry:
        conn_label = f"API Online [{telemetry.get('active_base', '8000').split(':')[-1]}]"
        dot_style = "background-color: #10b981; box-shadow: 0 0 8px #10b981;"
    else:
        conn_label = "Direct Engine Mode"
        dot_style = "background-color: #f59e0b; box-shadow: 0 0 8px #f59e0b;"

    st.markdown(
        f"""
        <div class="top-telemetry-bar">
            <div class="telemetry-indicator">
                <span class="dot-live" style="{dot_style}"></span>
                <span>{conn_label}</span>
            </div>
            <div class="telemetry-indicator">
                <span>Model: {provider_name} · <code>{model_name}</code></span>
            </div>
            <div class="telemetry-indicator">
                <span>Qdrant: Hybrid BM25+Dense</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero_section() -> None:
    """Renders the scholarly-yet-satirical SaaS hero header."""
    st.markdown(
        """
        <div class="hero-wrapper">
            <div class="meta-pill">
                Deterministic LangGraph Machine · Hybrid Vector Retrieval
            </div>
            <div class="hero-title">
                RealityCheck <span class="accent">AI</span>
            </div>
            <div class="hero-subtitle">
                An empirical, peer-reviewed evaluation engine comparing your career claims against
                machine-readability standards, Leland bullet formulas, and gold-standard MIT engineering benchmarks.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metrics_deck(roast: dict[str, Any], context_len: int, runtime_mode: str) -> None:
    """Renders summary KPI metric tiles."""
    overall_score = roast.get("overall_score", 0)
    jd_score = roast.get("jd_alignment_score", 0)
    ats_verdict = roast.get("ats_verdict", "FAIL")
    fixes_count = len(roast.get("actionable_fixes", []))

    # Dynamic styling
    score_color = "#10b981" if overall_score >= 75 else "#f59e0b" if overall_score >= 50 else "#f43f5e"
    verdict_color = "#10b981" if ats_verdict == "PASS" else "#f43f5e"

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">Overall Index</div>
                <div class="metric-value" style="color: {score_color};">{overall_score}<span style="font-size:1.1rem;color:#64748b;">/100</span></div>
                <div class="metric-foot">Calibrated Empirical Score</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">ATS Clearance</div>
                <div class="metric-value" style="color: {verdict_color};">{ats_verdict}</div>
                <div class="metric-foot">Machine Readability Filter</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">JD Alignment</div>
                <div class="metric-value" style="color: #60a5fa;">{jd_score}%</div>
                <div class="metric-foot">Semantic Keyword Overlap</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">Deficit Diagnostics</div>
                <div class="metric-value" style="color: #cbd5e1;">{fixes_count}</div>
                <div class="metric-foot">Ground-Truth Citations: {context_len}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
        <div style="text-align: right; margin-top: 0.5rem; margin-bottom: 1.5rem; font-size: 0.72rem; color: #64748b; font-family: 'JetBrains Mono', monospace;">
            Runtime Execution: {runtime_mode}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main Application Flow
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    # 1. Telemetry Top Bar & Hero
    render_telemetry_bar()
    render_hero_section()

    # 2. Session State Initialization
    if "evaluation_result" not in st.session_state:
        st.session_state.evaluation_result = None

    # 3. Dual-Column Input Architecture
    col_pdf, col_jd = st.columns([1, 1], gap="medium")

    with col_pdf:
        st.markdown(
            """
            <div class="saas-card">
                <div class="saas-card-header">
                    <span class="saas-card-title">Candidate Dossier</span>
                    <span class="saas-card-caption">PDF FORMAT ONLY</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        pdf_file = st.file_uploader(
            label="Candidate Resume PDF",
            type=["pdf"],
            help="Selectable-text resume PDF. Scanned image documents without OCR are unsupported.",
            label_visibility="collapsed",
        )
        if pdf_file:
            st.caption(f"Loaded: `{pdf_file.name}` ({round(pdf_file.size / 1024, 1)} KB)")

    with col_jd:
        st.markdown(
            """
            <div class="saas-card">
                <div class="saas-card-header">
                    <span class="saas-card-title">Target Opportunity Specification</span>
                    <span class="saas-card-caption">JOB DESCRIPTION</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        jd_default_placeholder = (
            "Paste target job description and engineering requirements here.\n\n"
            "Example:\n"
            "We are seeking a Distributed Systems Engineer with expertise in Python, RAG pipelines, "
            "asynchronous backends, vector indexes, and low-latency API architectures..."
        )
        jd_input = st.text_area(
            label="Target Job Description",
            height=165,
            placeholder=jd_default_placeholder,
            label_visibility="collapsed",
        )
        if jd_input:
            st.caption(f"Input Length: `{len(jd_input.strip())}` characters")

    # 4. Centralized Execution Button
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    is_ready = pdf_file is not None and len(jd_input.strip()) >= 20

    col_btn_l, col_btn_c, col_btn_r = st.columns([1, 2, 1])
    with col_btn_c:
        submit_btn = st.button(
            label="Initiate Empirical Roast",
            disabled=not is_ready,
            type="primary",
            use_container_width=True,
        )

        if not is_ready:
            st.markdown(
                """
                <div style="text-align: center; font-size: 0.75rem; color: #64748b; margin-top: 6px; font-family: 'JetBrains Mono', monospace;">
                    Provide a valid resume PDF and at least 20 characters of JD text to execute.
                </div>
                """,
                unsafe_allow_html=True,
            )

    # 5. UX State Management: Progressive Execution Logs
    if submit_btn and is_ready:
        st.session_state.evaluation_result = None
        pdf_bytes = pdf_file.getvalue()

        with st.status("Executing LangGraph State Machine Workflow...", expanded=True) as status_box:
            st.write("Ingesting candidate document stream and extracting raw text...")
            time.sleep(0.4)

            st.write("Node 1: Parsing resume schema (Work Experience, Education, Technical Skills)...")
            time.sleep(0.5)

            st.write("Node 2: Executing Qdrant Hybrid RAG query (Dense cosine + BM25 sparse vectors)...")
            st.write("Filtering metadata across ATS Guide, Leland Formulae, and MIT Resume Book...")
            time.sleep(0.6)

            st.write("Node 3: Synthesizing evidence-based critique and prescriptive remediation...")

            try:
                result = execute_evaluation(pdf_bytes, jd_input)
                st.session_state.evaluation_result = result
                status_box.update(
                    label="Evaluation Complete: Academic Critique Generated",
                    state="complete",
                    expanded=False,
                )
                st.toast("Evaluation finished successfully", icon="✅")
            except Exception as exc:
                status_box.update(
                    label=f"Pipeline Failure: {exc}",
                    state="error",
                    expanded=True,
                )
                st.error(f"Execution Error: {exc}")
                return

    # 6. Results Dashboard (Tabs Architecture)
    if st.session_state.evaluation_result:
        data = st.session_state.evaluation_result
        roast = data.get("final_roast", {})
        context = data.get("retrieved_context", [])
        retrieval_diagnostics = data.get("retrieval_diagnostics", {})
        parsed = data.get("parsed_sections", {})
        runtime_mode = data.get("execution_runtime", "Standard Pipeline")

        st.markdown("---")
        render_metrics_deck(roast, len(context), runtime_mode)

        # Tabbed Dashboard to Prevent Infinite Scrolling
        tab_roast, tab_fixes, tab_logs, tab_schema = st.tabs(
            [
                "The Roast",
                "Action Plan",
                "System Logs & RAG Evidence",
                "Extracted AST Schema",
            ]
        )

        # ── Tab 1: The Roast ─────────────────────────────────────────────────
        with tab_roast:
            brutal_text = roast.get("brutal_roast", "No narrative critique generated.")
            safe_text = html.escape(brutal_text)

            st.markdown(
                f"""
                <div class="roast-manifesto">
                    <span class="roast-quote-badge">Executive Pathologist Report</span>
                    <div class="roast-body">
                        "{safe_text}"
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # High-level ATS & JD breakdown callout
            c_ats, c_jd = st.columns(2)
            with c_ats:
                ats_state = roast.get("ats_verdict", "UNKNOWN")
                if ats_state == "PASS":
                    st.success("**ATS Parser Verdict: PASS**\nThe document conforms to machine-readable column and typeface standards.")
                else:
                    st.error("**ATS Parser Verdict: FAIL**\nStructural anomalies or non-standard formatting will result in automated system rejection.")

            with c_jd:
                jd_score = roast.get("jd_alignment_score", 0)
                st.info(f"**Target Role Alignment: {jd_score}%**\nSemantic overlap with target technical competencies and qualifications.")

        # ── Tab 2: Action Plan ───────────────────────────────────────────────
        with tab_fixes:
            fixes = roast.get("actionable_fixes", [])
            if fixes:
                # Severity sort priority
                sev_order = {"critical": 0, "major": 1, "minor": 2}
                fixes_sorted = sorted(fixes, key=lambda x: sev_order.get(x.get("severity", "major").lower(), 1))

                st.markdown(
                    f"<div style='font-size:0.85rem; color:#94a3b8; margin-bottom:1rem;'>Identified {len(fixes_sorted)} prescriptive defects mapped directly to vector knowledge-base guidelines.</div>",
                    unsafe_allow_html=True,
                )

                for idx, fix in enumerate(fixes_sorted, start=1):
                    sev = str(fix.get("severity", "major")).lower()
                    issue = html.escape(str(fix.get("issue", "Unspecified issue")))
                    fix_txt = html.escape(str(fix.get("fix", "No remediation specified")))
                    rule = html.escape(str(fix.get("rule_reference", "General best practice")))

                    st.markdown(
                        f"""
                        <div class="fix-entry {sev}">
                            <div class="fix-header">
                                <span class="fix-index">#{idx:02d}</span>
                                <span class="fix-issue-title">{issue}</span>
                                <span class="sev-badge {sev}">{sev}</span>
                            </div>
                            <div class="fix-rewrite">
                                <strong>Prescribed Revision:</strong> {fix_txt}
                            </div>
                            <div class="fix-citation">
                                <strong>Corpus Benchmark:</strong> "{rule}"
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                st.info("No actionable corrections were identified by the model.")

        # ── Tab 3: System Logs & RAG Evidence ────────────────────────────────
        with tab_logs:
            st.markdown(
                """
                <div style="font-size: 0.85rem; color: #94a3b8; margin-bottom: 0.75rem;">
                    <strong>Academic Evaluator Transparency Log:</strong> The chunks below were dynamically
                    retrieved from Qdrant using Hybrid Search (Dense Cosine Similarity + BM25 Sparse Keyword Vector)
                    and injected into the generation prompt.
                </div>
                """,
                unsafe_allow_html=True,
            )

            if context:
                ats_c = [c for c in context if c.get("doc_type") == "ats_rule"]
                wri_c = [c for c in context if c.get("doc_type") == "writing_rule"]
                mit_c = [c for c in context if c.get("doc_type") == "gold_example"]

                sub_ats, sub_wri, sub_mit, sub_raw = st.tabs(
                    [
                        f"ATS Guidelines ({len(ats_c)})",
                        f"Leland Bullet Formulas ({len(wri_c)})",
                        f"MIT Resumes ({len(mit_c)})",
                        "Raw Payload Inspector",
                    ]
                )

                def _render_chunk_list(chunks: list[dict[str, Any]]) -> None:
                    if not chunks:
                        st.caption("No chunks retrieved for this category.")
                        return
                    for i, chk in enumerate(chunks, start=1):
                        dtype = html.escape(str(chk.get("doc_type", "chunk")))
                        src = html.escape(str(chk.get("source", "source_file")))
                        pg = chk.get("page", "?")
                        score = chk.get("score", 0.0)
                        method = html.escape(str(chk.get("retrieval_method", "hybrid")))
                        raw_content = html.escape(str(chk.get("text", "")).strip())

                        st.markdown(
                            f"""
                            <div class="rag-chunk">
                                <div class="rag-chunk-meta">
                                    <span class="rag-tag {dtype}">{dtype}</span>
                                    <span>Source: {src} (p.{pg})</span>
                                    <span style="margin-left:auto; color:#60a5fa;">Score: {score} · {method}</span>
                                </div>
                                <div style="color: #cbd5e1; white-space: pre-wrap;">{raw_content}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                with sub_ats:
                    _render_chunk_list(ats_c)
                with sub_wri:
                    _render_chunk_list(wri_c)
                with sub_mit:
                    _render_chunk_list(mit_c)
                with sub_raw:
                    st.json(context)
            else:
                st.warning("No context chunks were returned by the retrieval node.")
                if retrieval_diagnostics:
                    st.json(retrieval_diagnostics)
                else:
                    st.info(
                        "The active backend did not return Qdrant diagnostics. "
                        "Restart the FastAPI worker so it loads the current code."
                    )

        # ── Tab 4: Extracted AST Schema (Node 1 Parser Output) ───────────────
        with tab_schema:
            st.markdown(
                """
                <div style="font-size: 0.85rem; color: #94a3b8; margin-bottom: 0.75rem;">
                    <strong>LangGraph Node 1 Output:</strong> Structured entities parsed from candidate PDF text.
                </div>
                """,
                unsafe_allow_html=True,
            )

            col_s1, col_s2 = st.columns(2)
            with col_s1:
                st.markdown("##### Extracted Work Experience")
                exp_list = parsed.get("work_experience", [])
                if exp_list:
                    for b in exp_list:
                        st.markdown(f"- {b}")
                else:
                    st.caption("No experience bullets detected.")

                st.markdown("##### Extracted Education")
                edu_list = parsed.get("education", [])
                if edu_list:
                    for e in edu_list:
                        st.markdown(f"- {e}")
                else:
                    st.caption("No education entries detected.")

            with col_s2:
                st.markdown("##### Technical Skills Identified")
                skills_list = parsed.get("skills", [])
                if skills_list:
                    pills = " ".join([f"`{s}`" for s in skills_list])
                    st.markdown(pills)
                else:
                    st.caption("No specific technical skills extracted.")

                summary_str = parsed.get("summary", "")
                if summary_str:
                    st.markdown("##### Candidate Summary")
                    st.info(summary_str)


if __name__ == "__main__":
    main()
