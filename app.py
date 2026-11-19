import uuid
import requests
import streamlit as st

import os
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="Research Intelligence Agent",
    page_icon="🔬",
    layout="wide",
)

st.title("Research Intelligence Agent")
st.caption("Autonomous Graph-RAG Literature Synthesis • LangGraph + FAISS + Neo4j + Gemini/Groq")

# Sidebar: System Status, PDF Ingestion & Session Controls
with st.sidebar:
    st.header("System Telemetry")
    try:
        health = requests.get(f"{API_BASE_URL}/health", timeout=3).json()
        st.success("FastAPI Backend: Online")
        st.metric("FAISS Indexed Chunks", health["storage"]["faiss_vectors"])
        neo4j_status = health["storage"]["neo4j"]["status"]
        st.write(f"**Neo4j Graph:** {neo4j_status}")
        st.write(f"**Primary LLM:** {health['models']['primary_llm']}")
        st.write(f"**Fast Extractor:** {health['models']['fast_llm']}")
    except Exception:
        st.error("FastAPI Backend: Offline")
        st.info("Start backend with:\nuvicorn api.main:app --reload")

    st.divider()
    st.header("Add Paper to Knowledge Base")
    uploaded_pdf = st.file_uploader("Upload Research PDF", type=["pdf"])
    if uploaded_pdf is not None and st.button("Index PDF into FAISS"):
        with st.spinner("Extracting & embedding PDF..."):
            files = {"file": (uploaded_pdf.name, uploaded_pdf.getvalue(), "application/pdf")}
            resp = requests.post(f"{API_BASE_URL}/upload-pdf", files=files, timeout=30)
            if resp.ok:
                data = resp.json()
                st.success(f"Indexed {data['chunks_indexed']} chunks from {data['filename']} ({data['pages']} pages)")
            else:
                st.error(resp.text)

    st.divider()
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = f"ui-{uuid.uuid4().hex[:8]}"
    st.text_input("Session Thread ID", key="thread_id")
    if st.button("New Session Thread"):
        st.session_state.thread_id = f"ui-{uuid.uuid4().hex[:8]}"
        st.rerun()

# Main Research Workspace
objective = st.text_area(
    "Research Objective",
    value="Compare deep learning and transformer architectures for fake news detection, including benchmark datasets and key limitations.",
    height=90,
)

col1, col2 = st.columns([1, 4])
with col1:
    max_iters = st.slider("Max Research Loops", min_value=1, max_value=5, value=3)
with col2:
    run_clicked = st.button("Run Autonomous Research", type="primary", use_container_width=True)

if run_clicked and objective.strip():
    with st.spinner("Agent is planning subtasks, querying FAISS/Neo4j/arXiv, and verifying citations..."):
        try:
            response = requests.post(
                f"{API_BASE_URL}/research",
                json={
                    "objective": objective.strip(),
                    "max_iterations": max_iters,
                    "thread_id": st.session_state.thread_id,
                },
                timeout=120,
            )
            if not response.ok:
                st.error(f"API Error ({response.status_code}): {response.text}")
            else:
                st.session_state.last_result = response.json()
        except Exception as exc:
            st.error(f"Could not connect to FastAPI backend: {exc}")

if "last_result" in st.session_state:
    res = st.session_state.last_result

    # Execution Trace & Plan
    with st.expander("Agent Execution Trace & Subtask Plan", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Decomposed Subtasks")
            for step in res.get("plan", []):
                st.markdown(f"**{step['step_id']}.** {step['question']} ({step['recommended_tool']})")
        with c2:
            st.subheader("LangGraph Node Logs")
            for line in res.get("logs", []):
                st.code(line, language="text")

    # Synthesized Report
    st.subheader("Synthesized Literature Report")
    st.markdown(res["final_report"])

    st.download_button(
        label="Download Report (.md)",
        data=res["final_report"],
        file_name=f"research_report_{res['thread_id']}.md",
        mime="text/markdown",
    )

    # Structured Citation Inspector
    st.subheader(f"Grounded Evidence Cards ({len(res.get('evidence', []))} Citations)")
    for item in res.get("evidence", []):
        with st.expander(f"{item['citation_id']} {item['title']} [{item['source_type']}]"):
            st.write(f"**Paper ID:** {item['paper_id']}")
            if item.get("url"):
                st.write(f"**URL:** {item['url']}")
            st.write(f"**Methods:** {', '.join(item.get('methods', [])) or 'Not specified'}")
            st.write(f"**Datasets:** {', '.join(item.get('datasets', [])) or 'Not specified'}")
            st.write(f"**Key Finding:** {item.get('snippet', '')}")
            if item.get("limitations"):
                st.write(f"**Limitations:** {item['limitations']}")
