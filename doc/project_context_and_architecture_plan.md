# Project Context: Roast My Resume (RAG Edition)

## 1. Project Overview
This project is an AI-powered, backend-heavy "Roast My Resume" application built to satisfy academic computer science lab requirements. It evaluates a user's resume against a target Job Description (JD) using an advanced Retrieval-Augmented Generation (RAG) architecture. 

Instead of generating generic insults, the system performs an **evidence-based critique** by comparing the user's resume against industry-standard ATS guidelines, bullet-point writing formulas, and gold-standard MIT engineering resumes.

## 2. Technical Stack
*   **Backend API:** FastAPI (Python)
*   **AI Orchestration:** LangGraph (Stateful workflow) & LangChain
*   **Vector Database:** Qdrant (Utilized for Hybrid Search: Sparse BM25 Keywords + Dense Vectors)
*   **LLM & Embeddings:** OpenAI or local HuggingFace equivalents
*   **Frontend UI:** Streamlit (Minimalist, demo-friendly Python UI)

## 3. Ground Truth Dataset (Vector Store Context)
The RAG knowledge base is built from three specific PDF documents. These act as the "ground truth" the LLM uses to evaluate right vs. wrong.

1.  **`ApplicantTrackingSystem-Guide_2.pdf`**
    *   **Content:** Strict rules on machine readability, acceptable fonts, date formatting, and structure.
    *   **Utility:** Used to catch structural flaws (e.g., using tables, weird date formats) that cause ATS rejection.
2.  **`Leland—The-Ultimate-Career-Resume-Guide+Examples_removed_2.pdf`**
    *   **Content:** Industry-standard writing rules, specifically Google's XYZ formula ("Accomplished X, as measured by Y, by doing Z"), action verbs, and impact quantification.
    *   **Utility:** Used to critique weak bullet points and suggest highly specific phrasing improvements.
3.  **`Resume-Book_removed.pdf`**
    *   **Content:** A curated collection of 9 top-tier MIT technical/engineering resumes.
    *   **Utility:** Provides semantic benchmarks. When a user claims "managed servers," RAG retrieves how an MIT student phrased a similar engineering accomplishment.

## 4. Data Ingestion & Pipeline Strategy
To ensure high accuracy, we will use **Hybrid Search (Metadata Filtering + Dense Vectors)** in Qdrant. 

*   **Chunking & Tagging:** When processing the PDFs, chunks will be tagged with metadata. 
    *   *Example Tags:* `{"doc_type": "rule", "section": "format"}`, `{"doc_type": "gold_example", "section": "experience"}`.
*   **Why?** When the pipeline analyzes the user's "Work Experience" section, Qdrant will apply a metadata filter to strictly search for "experience" chunks, eliminating hallucinations caused by retrieving irrelevant rules.

## 5. LangGraph Execution Flow
The core engine is a linear LangGraph state machine. It avoids complex cycles to remain robust for demo purposes.

*   **Graph State (`TypedDict`):** Tracks `resume_text`, `parsed_sections`, `job_description`, `retrieved_context`, and `roast_output`.
*   **Node 1: Parser (`node_parse_resume`)**: Uses an LLM tool-calling/structured output chain to convert the raw user PDF text into structured JSON (Experience, Skills, Education).
*   **Node 2: Hybrid Retriever (`node_retrieve_context`)**: Takes the extracted sections and JD, constructs metadata-filtered queries, and pulls the top rules and examples from Qdrant.
*   **Node 3: Generator (`node_generate_roast`)**: Feeds the original resume, JD, and Qdrant context into the final prompt to generate the "Brutal Roast" and "Actionable Fixes".

## 6. Frontend (Streamlit) & Demo Experience
The UI is purposefully thin but designed to impress an academic evaluator. 
*   **Inputs:** Sidebar containing a file uploader (PDF) and text area (Job Description).
*   **Execution UI:** A dynamic loading spinner that updates as the LangGraph state progresses (e.g., "Extracting text...", "Querying Vector Database...", "Synthesizing Roast...").
*   **Output:** The generated Roast and Fixes.
*   **Evaluator Transparency:** An expandable UI component (`st.expander("🔍 View RAG Retrieval Context")`) that displays the exact database chunks retrieved by Qdrant. This proves the RAG architecture is functional and not faked.

## 7. Next Immediate Steps
1.  Initialize Python virtual environment.
2.  Install `requirements.txt` (FastAPI, LangChain, LangGraph, Qdrant-client, Streamlit, PyPDF2).
3.  Implement `ingest.py` to chunk, tag, and embed the 3 PDF documents into Qdrant.
4.  Implement `main.py` containing the LangGraph logic and FastAPI endpoints.
5.  Implement `app.py` for the Streamlit frontend.