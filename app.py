"""
app.py — Streamlit Frontend for "Roast My Resume".

Design goals
────────────
- Academically polished, minimal design with clear section hierarchy
- Visual LangGraph node progression during processing
- Evidence-first output: every fix is anchored to a retrieved context chunk
- Evaluator transparency: full RAG retrieval context visible in an expander
- Responsive layout using Streamlit columns

Usage
─────
    streamlit run app.py
"""

from __future__ import annotations

import time
from typing import Any

import httpx
import streamlit as st

# Import provider info helper (reads .env directly in the UI process)
try:
    from llm_factory import provider_info as _provider_info
    from config import get_settings as _get_settings
    _LLM_INFO = _provider_info()
    _LLM_CFG  = _get_settings()
except Exception:
    _LLM_INFO = {}
    _LLM_CFG  = None

# ─────────────────────────────────────────────────────────────────────────────
# Page configuration (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Roast My Resume — RAG Edition",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

API_PORT = getattr(_LLM_CFG, "api_port", 8000) if _LLM_CFG else 8000
# Prioritize 127.0.0.1 on Windows to bypass IPv6 localhost loopback collisions
CANDIDATE_BASES = [
    f"http://127.0.0.1:{API_PORT}",
    f"http://localhost:{API_PORT}",
]
EVALUATE_ENDPOINT = f"{CANDIDATE_BASES[0]}/api/evaluate"
HEALTH_ENDPOINT = f"{CANDIDATE_BASES[0]}/api/health"
COLLECTION_ENDPOINT = f"{CANDIDATE_BASES[0]}/api/collection-info"

SEVERITY_COLOR = {
    "critical": "🔴",
    "major":    "🟡",
    "minor":    "🟢",
}
DOC_TYPE_BADGE = {
    "ats_rule":     ("📋", "#1a3a5c", "ATS Rule"),
    "writing_rule": ("✍️",  "#3a2a1c", "Writing Rule"),
    "gold_example": ("⭐", "#1c3a1c", "MIT Gold Example"),
}

# ─────────────────────────────────────────────────────────────────────────────
# Styling
# ─────────────────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
<style>
/* ── Global ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Score ring ── */
.score-ring {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    width: 130px;
    height: 130px;
    border-radius: 50%;
    border: 6px solid;
    margin: auto;
    font-weight: 700;
}
.score-ring .score-num {
    font-size: 2.4rem;
    line-height: 1;
}
.score-ring .score-label {
    font-size: 0.65rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* ── Fix card ── */
.fix-card {
    border-left: 4px solid;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin: 10px 0;
    background: rgba(255,255,255,0.03);
}
.fix-card.critical { border-color: #ff4b4b; }
.fix-card.major    { border-color: #ffa500; }
.fix-card.minor    { border-color: #21c55d; }

/* ── Context chunk ── */
.chunk-card {
    border-radius: 8px;
    padding: 10px 14px;
    margin: 8px 0;
    font-size: 0.85rem;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1.6;
}

/* ── Node progress badges ── */
.node-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.04em;
}
.node-done    { background:#1a4731; color:#21c55d; }
.node-active  { background:#1a3060; color:#60a5fa; }
.node-pending { background:#2a2a2a; color:#6b7280; }

/* ── Metric tiles ── */
.metric-tile {
    text-align: center;
    padding: 16px 8px;
    border-radius: 10px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
}
.metric-tile .mt-value {
    font-size: 2rem;
    font-weight: 700;
}
.metric-tile .mt-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #9ca3af;
    margin-top: 4px;
}

/* ── Header ── */
.page-header {
    text-align: center;
    padding: 24px 0 12px;
}
.page-header h1 {
    font-size: 2.6rem;
    font-weight: 700;
    background: linear-gradient(135deg, #ff4b4b 0%, #ff8c42 50%, #ffd166 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 4px;
}
.page-header p {
    color: #9ca3af;
    font-size: 1rem;
}

/* ── Verdict badge ── */
.verdict-pass { color: #21c55d; font-weight: 700; font-size: 1.2rem; }
.verdict-fail { color: #ff4b4b; font-weight: 700; font-size: 1.2rem; }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────


def check_backend_health() -> dict[str, Any] | None:
    """Ping the FastAPI health endpoint across candidate URLs."""
    for base in CANDIDATE_BASES:
        try:
            r = httpx.get(f"{base}/api/health", timeout=3)
            if r.status_code == 200:
                data = r.json()
                data["active_base"] = base
                return data
        except Exception:
            pass
    return None


def call_evaluate(pdf_bytes: bytes, job_description: str) -> dict[str, Any]:
    """
    Evaluate the resume PDF against the job description.
    Tries the FastAPI REST backend first; if offline or returning 404
    (e.g. port conflict with another application), seamlessly falls back
    to in-process LangGraph pipeline execution so the application is always functional.
    """
    # 1. Try FastAPI endpoints
    for base in CANDIDATE_BASES:
        endpoint = f"{base}/api/evaluate"
        try:
            with httpx.Client(timeout=300) as client:
                response = client.post(
                    endpoint,
                    files={"resume_pdf": ("resume.pdf", pdf_bytes, "application/pdf")},
                    data={"job_description": job_description},
                )
            if response.status_code == 200:
                res_data = response.json()
                res_data["execution_mode"] = f"FastAPI REST API ({base})"
                return res_data
        except Exception:
            pass

    # 2. In-process LangGraph fallback
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
        "execution_mode": "Direct LangGraph Engine (In-Process Fallback)",
        "final_roast": final_state.get("final_roast", {}),
        "retrieved_context": final_state.get("retrieved_context", []),
        "parsed_sections": final_state.get("parsed_sections", {}),
        "error": final_state.get("error"),
    }


def score_color(score: int) -> str:
    if score >= 75:
        return "#21c55d"
    elif score >= 50:
        return "#ffa500"
    else:
        return "#ff4b4b"


def render_score_ring(score: int, label: str) -> str:
    color = score_color(score)
    return f"""
    <div class="score-ring" style="border-color:{color}; color:{color};">
        <div class="score-num">{score}</div>
        <div class="score-label">{label}</div>
    </div>
    """


def render_node_progress(active_node: str) -> None:
    """Render the 3-node pipeline progress indicator."""
    nodes = [
        ("node_parse",    "① Parsing Resume",   "📄"),
        ("node_retrieve", "② Hybrid RAG Query", "🔍"),
        ("node_generate", "③ Generating Roast", "🔥"),
    ]
    order = [n[0] for n in nodes]
    active_idx = order.index(active_node) if active_node in order else -1

    cols = st.columns(3)
    for i, (node_id, label, icon) in enumerate(nodes):
        if i < active_idx:
            css_class = "node-done"
            status_str = "✓ Done"
        elif i == active_idx:
            css_class = "node-active"
            status_str = "● Running"
        else:
            css_class = "node-pending"
            status_str = "○ Pending"

        with cols[i]:
            st.markdown(
                f'<div style="text-align:center;">'
                f'<div class="node-badge {css_class}">{icon} {label}</div>'
                f'<div style="font-size:0.72rem;color:#6b7280;margin-top:4px;">{status_str}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )


def render_fix_card(fix: dict[str, Any], index: int) -> None:
    """Render a single actionable fix card."""
    severity = fix.get("severity", "major").lower()
    emoji = SEVERITY_COLOR.get(severity, "🟡")
    issue = fix.get("issue", "")
    fix_text = fix.get("fix", "")
    rule_ref = fix.get("rule_reference", "")

    badge_color = {"critical": "#ff4b4b", "major": "#ffa500", "minor": "#21c55d"}.get(severity, "#ffa500")

    st.markdown(
        f"""
        <div class="fix-card {severity}">
            <div style="display:flex; align-items:center; gap:8px; margin-bottom:6px;">
                <span style="font-size:1.1rem;">{emoji}</span>
                <span style="font-weight:600; font-size:0.95rem;">Fix #{index}: {issue}</span>
                <span style="margin-left:auto; font-size:0.7rem; padding:2px 8px;
                             border-radius:12px; background:{badge_color}22; color:{badge_color};
                             font-weight:600; text-transform:uppercase;">{severity}</span>
            </div>
            <div style="color:#d1d5db; font-size:0.88rem; margin-bottom:6px;">
                <strong>→ </strong>{fix_text}
            </div>
            <div style="color:#6b7280; font-size:0.78rem; font-style:italic;
                        border-top:1px solid rgba(255,255,255,0.06); padding-top:6px;">
                📚 Rule: "{rule_ref[:200]}{"…" if len(rule_ref) > 200 else ""}"
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_context_chunk(chunk: dict[str, Any], index: int) -> None:
    """Render a single Qdrant retrieval result."""
    doc_type = chunk.get("doc_type", "unknown")
    icon, bg_color, label = DOC_TYPE_BADGE.get(doc_type, ("📄", "#2a2a2a", doc_type))
    source = chunk.get("source", "unknown")
    page = chunk.get("page", "?")
    score = chunk.get("score", 0)
    method = chunk.get("retrieval_method", "?")
    text = chunk.get("text", "")

    st.markdown(
        f"""
        <div class="chunk-card" style="background:{bg_color}55; border:1px solid {bg_color}88;">
            <div style="display:flex; align-items:center; gap:6px; margin-bottom:6px; flex-wrap:wrap;">
                <span style="font-size:1rem;">{icon}</span>
                <span style="font-weight:600; font-size:0.8rem; color:#e5e7eb;">[{index}] {label}</span>
                <span style="font-size:0.72rem; color:#9ca3af;">| {source} | p.{page}</span>
                <span style="margin-left:auto; font-size:0.7rem; color:#60a5fa;">
                    score={score} | {method}
                </span>
            </div>
            <div style="color:#d1d5db; white-space:pre-wrap; font-size:0.8rem;">{text[:500]}{"…" if len(text) > 500 else ""}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────


def render_sidebar() -> tuple[bytes | None, str | None]:
    """Render the sidebar and return (pdf_bytes, job_description)."""
    with st.sidebar:
        st.markdown("## 🔥 Roast My Resume")
        st.markdown("*Evidence-based critique powered by Hybrid RAG*")
        st.divider()

        # ── Backend health indicator ────────────────────────────────────────
        health = check_backend_health()
        if health:
            qdrant_ok  = health.get("qdrant_connected", False)
            coll_ok    = health.get("qdrant_collection_exists", False)
            provider   = health.get("llm_provider", "?")

            # Provider colour coding
            _provider_colors = {"openai": "#74aa9c", "groq": "#f55036", "ollama": "#a78bfa"}
            _pcolor = _provider_colors.get(provider, "#9ca3af")

            st.markdown("**System Status**")
            st.markdown(
                f"""
                <div style="display:grid; grid-template-columns:1fr 1fr; gap:6px; font-size:0.82rem;">
                    <div>{'🟢' if qdrant_ok else '🔴'} Qdrant DB</div>
                    <div>{'🟢' if coll_ok else '🟡'} Collection</div>
                    <div>🟢 FastAPI</div>
                    <div>
                        <span style="background:{_pcolor}22; color:{_pcolor};
                                     padding:1px 7px; border-radius:10px;
                                     font-weight:600; font-size:0.75rem;">
                            {provider.upper()}
                        </span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # ── Active LLM config panel ─────────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            _info = _LLM_INFO
            if _info:
                key_ok = _info.get("key_set", False)
                st.markdown(
                    f"""
                    <div style="background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08);
                                border-radius:8px; padding:10px 12px; font-size:0.78rem; line-height:1.8;">
                        <div style="font-weight:600; margin-bottom:4px; color:#e5e7eb;">
                            🤖 {_info.get('provider', provider)} Config
                        </div>
                        <div style="color:#9ca3af;">
                            Chat: <code style="color:#60a5fa;">{_info.get('chat_model','?')}</code>
                        </div>
                        <div style="color:#9ca3af;">
                            Embed: <code style="color:#a78bfa;">{_info.get('embed_model','?')}</code>
                        </div>
                        <div style="color:#9ca3af;">
                            Embed via: <em>{_info.get('embed_backend','?')}</em>
                        </div>
                        <div style="margin-top:4px;">
                            {'🟢 API Key set' if key_ok else '🔴 API Key missing — check .env'}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.warning(
                f"⚠️ FastAPI is offline on port {API_PORT}.\n\n"
                "⚡ **Direct LangGraph Mode is active!** Evaluation will run seamlessly in-process."
            )
            _info = _LLM_INFO
            if _info:
                key_ok = _info.get("key_set", False)
                provider = _info.get("provider", "LLM")
                st.markdown(
                    f"""
                    <div style="background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08);
                                border-radius:8px; padding:10px 12px; font-size:0.78rem; line-height:1.8;">
                        <div style="font-weight:600; margin-bottom:4px; color:#e5e7eb;">
                            🤖 {provider} Config
                        </div>
                        <div style="color:#9ca3af;">
                            Chat: <code style="color:#60a5fa;">{_info.get('chat_model','?')}</code>
                        </div>
                        <div style="margin-top:4px;">
                            {'🟢 API Key set' if key_ok else '🔴 API Key missing in .env'}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.divider()

        # ── Resume upload ───────────────────────────────────────────────────
        st.markdown("### 📄 Upload Resume")
        pdf_file = st.file_uploader(
            label="Resume PDF",
            type=["pdf"],
            help="Upload a text-selectable PDF (not a scanned image).",
            label_visibility="collapsed",
        )

        st.divider()

        # ── Job Description ─────────────────────────────────────────────────
        st.markdown("### 💼 Job Description")
        jd_placeholder = (
            "Paste the full job description here.\n\n"
            "Example:\n"
            "We are looking for a Senior Software Engineer with 5+ years of "
            "experience in Python, distributed systems, and cloud infrastructure (AWS/GCP)..."
        )
        jd_text = st.text_area(
            label="Job Description",
            height=280,
            placeholder=jd_placeholder,
            label_visibility="collapsed",
        )

        st.divider()

        # ── Submit button ───────────────────────────────────────────────────
        submit = st.button(
            "🔥 Roast My Resume",
            type="primary",
            use_container_width=True,
            disabled=(pdf_file is None or not jd_text.strip()),
        )

        if pdf_file is None:
            st.caption("⬆️ Upload a PDF to enable evaluation.")
        elif not jd_text.strip():
            st.caption("✏️ Add a job description to enable evaluation.")

        st.divider()

        # ── Tip for evaluators ──────────────────────────────────────────────
        with st.expander("ℹ️ Academic Evaluator Notes"):
            st.markdown(
                """
**Architecture Overview**

| Component | Technology |
|-----------|-----------|
| Backend   | FastAPI + Pydantic v2 |
| Orchestration | LangGraph (StateGraph) |
| Vector DB | Qdrant Hybrid Search |
| Retrieval | Dense cosine + BM25 Sparse |
| LLM       | Groq (default) / OpenAI / Ollama |

**LLM Providers**
- `groq` — Free cloud API, ultra-fast LLaMA/Mixtral inference
- `openai` — GPT-4o-mini / GPT-4o via OpenAI API
- `ollama` — Fully local GPU execution (no API key)

**Ground-Truth Knowledge Base**
- `ats_rule` — ATS formatting rules
- `writing_rule` — Leland XYZ bullet formulas
- `gold_example` — MIT engineering resumes

The **🔍 System Logs** expander shows exact Qdrant chunks used for generation.
                """
            )

    if submit and pdf_file and jd_text.strip():
        return pdf_file.getvalue(), jd_text.strip()
    return None, None


# ─────────────────────────────────────────────────────────────────────────────
# Main results view
# ─────────────────────────────────────────────────────────────────────────────


def render_results(data: dict[str, Any]) -> None:
    """Render the complete evaluation results."""
    roast     = data.get("final_roast", {})
    context   = data.get("retrieved_context", [])
    parsed    = data.get("parsed_sections", {})

    overall_score   = roast.get("overall_score", 0)
    jd_align        = roast.get("jd_alignment_score", 0)
    brutal_roast    = roast.get("brutal_roast", "")
    fixes           = roast.get("actionable_fixes", [])
    ats_verdict     = roast.get("ats_verdict", "UNKNOWN")

    # ── Score dashboard ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📊 Evaluation Dashboard")
    exec_mode = data.get("execution_mode", "LangGraph RAG Engine")
    st.caption(f"⚡ Mode: `{exec_mode}`")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(render_score_ring(overall_score, "OVERALL"), unsafe_allow_html=True)

    with col2:
        st.markdown(render_score_ring(jd_align, "JD ALIGN"), unsafe_allow_html=True)

    with col3:
        verdict_class = "verdict-pass" if ats_verdict == "PASS" else "verdict-fail"
        verdict_icon  = "✅" if ats_verdict == "PASS" else "❌"
        st.markdown(
            f'<div class="metric-tile" style="margin-top:8px;">'
            f'  <div class="mt-value">'
            f'    <span class="{verdict_class}">{verdict_icon} {ats_verdict}</span>'
            f'  </div>'
            f'  <div class="mt-label">ATS Verdict</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f'<div class="metric-tile" style="margin-top:8px;">'
            f'  <div class="mt-value" style="color:#60a5fa;">{len(fixes)}</div>'
            f'  <div class="mt-label">Actionable Fixes</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Brutal Roast narrative ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔥 The Verdict")

    roast_severity = "error" if overall_score < 50 else "warning" if overall_score < 75 else "info"
    if roast_severity == "error":
        st.error(f"**Brutal Roast:**\n\n{brutal_roast}", icon="🔥")
    elif roast_severity == "warning":
        st.warning(f"**Brutal Roast:**\n\n{brutal_roast}", icon="⚠️")
    else:
        st.info(f"**Critique:**\n\n{brutal_roast}", icon="💡")

    # ── Actionable Fixes ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🛠️ Actionable Fixes")

    if fixes:
        # Sort: critical first
        severity_order = {"critical": 0, "major": 1, "minor": 2}
        fixes_sorted = sorted(fixes, key=lambda x: severity_order.get(x.get("severity", "major"), 1))

        for i, fix in enumerate(fixes_sorted, 1):
            render_fix_card(fix, i)
    else:
        st.info("No specific fixes were generated.")

    # ── Parsed sections (collapsible) ─────────────────────────────────────
    st.markdown("---")
    with st.expander("📑 Parsed Resume Sections (LangGraph Node 1 Output)", expanded=False):
        tab_exp, tab_edu, tab_skills = st.tabs(["Work Experience", "Education", "Skills"])

        with tab_exp:
            bullets = parsed.get("work_experience", [])
            if bullets:
                for b in bullets:
                    st.markdown(f"- {b}")
            else:
                st.caption("No experience bullets extracted.")

        with tab_edu:
            edu = parsed.get("education", [])
            if edu:
                for e in edu:
                    st.markdown(f"- {e}")
            else:
                st.caption("No education entries extracted.")

        with tab_skills:
            skills = parsed.get("skills", [])
            if skills:
                st.markdown(" · ".join([f"`{s}`" for s in skills]))
            else:
                st.caption("No skills extracted.")

    # ── RAG Retrieval Context (Evaluator Feature) ──────────────────────────
    st.markdown("---")
    with st.expander("🔍 System Logs: RAG Retrieval Context (Qdrant Evidence)", expanded=False):
        st.markdown(
            f"""
            > **Evaluator Note:** The following {len(context)} text chunks were retrieved from
            > the Qdrant vector database using **Hybrid Search** (dense cosine similarity +
            > BM25 sparse keyword matching). These exact chunks were injected into the LLM
            > prompt to generate the evidence-based critique above.
            """
        )

        if context:
            # Group by doc_type
            ats_chunks     = [c for c in context if c.get("doc_type") == "ats_rule"]
            writing_chunks = [c for c in context if c.get("doc_type") == "writing_rule"]
            gold_chunks    = [c for c in context if c.get("doc_type") == "gold_example"]

            tabs = st.tabs([
                f"📋 ATS Rules ({len(ats_chunks)})",
                f"✍️ Writing Rules ({len(writing_chunks)})",
                f"⭐ MIT Gold Examples ({len(gold_chunks)})",
            ])

            with tabs[0]:
                if ats_chunks:
                    for i, chunk in enumerate(ats_chunks, 1):
                        render_context_chunk(chunk, i)
                else:
                    st.caption("No ATS rule chunks retrieved.")

            with tabs[1]:
                if writing_chunks:
                    for i, chunk in enumerate(writing_chunks, 1):
                        render_context_chunk(chunk, i)
                else:
                    st.caption("No writing rule chunks retrieved.")

            with tabs[2]:
                if gold_chunks:
                    for i, chunk in enumerate(gold_chunks, 1):
                        render_context_chunk(chunk, i)
                else:
                    st.caption("No gold example chunks retrieved.")

            # Raw JSON for technical evaluators
            with st.expander("🗄️ Raw JSON (all retrieved chunks)", expanded=False):
                st.json(context)
        else:
            st.warning("No retrieval context was captured. Is Qdrant populated? Run `ingest.py`.")


# ─────────────────────────────────────────────────────────────────────────────
# Main application
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    # ── Page header ───────────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="page-header">
            <h1>🔥 Roast My Resume</h1>
            <p>Evidence-based AI critique · Hybrid RAG · LangGraph · Qdrant</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Sidebar → get inputs ──────────────────────────────────────────────────
    pdf_bytes, job_description = render_sidebar()

    # ── Landing / Instruction state ───────────────────────────────────────────
    if "result" not in st.session_state:
        st.session_state.result = None

    if pdf_bytes is None and st.session_state.result is None:
        _render_landing()
        return

    # ── Processing ────────────────────────────────────────────────────────────
    if pdf_bytes is not None:
        st.session_state.result = None  # clear previous result
        _run_pipeline(pdf_bytes, job_description)

    # ── Results ───────────────────────────────────────────────────────────────
    if st.session_state.result is not None:
        render_results(st.session_state.result)


def _render_landing() -> None:
    """Render the welcome / instruction view."""
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            """
            <div style="text-align:center; padding:40px 20px; border:1px solid rgba(255,255,255,0.08);
                         border-radius:16px; background:rgba(255,255,255,0.02);">
                <div style="font-size:4rem; margin-bottom:16px;">🔥</div>
                <h2 style="font-weight:700; margin-bottom:8px;">Ready to get roasted?</h2>
                <p style="color:#9ca3af; font-size:0.95rem;">
                    Upload your resume PDF and paste a job description in the sidebar to begin.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br><br>", unsafe_allow_html=True)

    # Feature overview cards
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """
            <div class="metric-tile">
                <div style="font-size:2rem;">🧠</div>
                <div style="font-weight:600; margin:8px 0 4px;">Hybrid RAG</div>
                <div style="color:#9ca3af; font-size:0.8rem;">
                    Dense cosine similarity + BM25 sparse keyword search
                    against 3 ground-truth documents.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="metric-tile">
                <div style="font-size:2rem;">⚙️</div>
                <div style="font-weight:600; margin:8px 0 4px;">LangGraph Pipeline</div>
                <div style="color:#9ca3af; font-size:0.8rem;">
                    3-node deterministic state machine: Parse → Retrieve → Generate.
                    No hallucination loops.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """
            <div class="metric-tile">
                <div style="font-size:2rem;">🔍</div>
                <div style="font-weight:600; margin:8px 0 4px;">Full Transparency</div>
                <div style="color:#9ca3af; font-size:0.8rem;">
                    Every critique is backed by exact retrieved chunks
                    visible in the System Logs expander.
                    Powered by Groq / OpenAI / Ollama.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _run_pipeline(pdf_bytes: bytes, job_description: str) -> None:
    """Execute the API call with LangGraph node progress UI."""

    node_labels = [
        ("node_parse",    "📄 Parsing Resume",              "Extracting structured sections from your PDF …"),
        ("node_retrieve", "🔍 Querying Qdrant (Hybrid RAG)", "Running dense + BM25 search against ground-truth docs …"),
        ("node_generate", "🔥 Generating Roast",             "Synthesising evidence-based critique …"),
    ]

    progress_placeholder = st.empty()
    status_placeholder   = st.empty()

    # Simulate node-by-node progress while the API processes
    with st.spinner(""):
        for i, (node_id, label, desc) in enumerate(node_labels):
            with progress_placeholder.container():
                st.markdown("#### ⚙️ LangGraph Pipeline Running")
                render_node_progress(node_id)
            status_placeholder.info(f"**{label}** — {desc}")

            if i < len(node_labels) - 1:
                time.sleep(0.8)  # brief visual pause between node labels

        # Execute evaluation (API first, direct LangGraph engine fallback)
        try:
            result = call_evaluate(pdf_bytes, job_description)

            # Mark all nodes done
            with progress_placeholder.container():
                st.markdown("#### ✅ LangGraph Pipeline Complete")
                cols = st.columns(3)
                done_nodes = [
                    ("📄 Parsing Resume",              "✓ Done"),
                    ("🔍 Querying Qdrant (Hybrid RAG)", "✓ Done"),
                    ("🔥 Generating Roast",             "✓ Done"),
                ]
                for col, (lbl, stat) in zip(cols, done_nodes):
                    with col:
                        st.markdown(
                            f'<div style="text-align:center;">'
                            f'<div class="node-badge node-done">{lbl}</div>'
                            f'<div style="font-size:0.72rem;color:#21c55d;margin-top:4px;">{stat}</div>'
                            f"</div>",
                            unsafe_allow_html=True,
                        )

            mode = result.get("execution_mode", "LangGraph")
            status_placeholder.success(f"✅ Evaluation complete ({mode})! Results displayed below.")
            time.sleep(1.0)
            progress_placeholder.empty()
            status_placeholder.empty()

            st.session_state.result = result

        except Exception as exc:
            progress_placeholder.empty()
            status_placeholder.empty()
            st.error(f"**Evaluation error:** {exc}")
            st.session_state.result = None


if __name__ == "__main__":
    main()

