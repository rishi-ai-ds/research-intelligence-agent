import os
import json
import uuid
import requests
import streamlit as st
import streamlit.components.v1 as components

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="Research Intelligence Agent",
    page_icon="??",
    layout="wide",
)

st.title("Research Intelligence Agent")
st.caption("Autonomous Graph-RAG Literature Synthesis ? LangGraph + FAISS + Neo4j + Gemini/Groq")


def render_interactive_kg(graph_data: dict):
    nodes = graph_data.get("nodes", [])
    edges = graph_data.get("edges", [])
    if not nodes:
        st.info("Neo4j is connected, but no nodes exist yet. Run a research query to populate Papers, Methods, and Datasets!")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Papers (Blue)", sum(1 for n in nodes if n["group"] == "Paper"))
    c2.metric("Methods (Green)", sum(1 for n in nodes if n["group"] == "Method"))
    c3.metric("Datasets (Orange)", sum(1 for n in nodes if n["group"] == "Dataset"))
    c4.metric("Relationships", len(edges))

    nodes_json = json.dumps(nodes)
    edges_json = json.dumps(edges)

    html_canvas = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
      <style>
        body {{ margin: 0; background-color: #0e1117; font-family: sans-serif; color: #e2e8f0; }}
        #kg-wrapper {{
          position: relative;
          width: 100%;
          height: 560px;
          background-color: #0e1117;
          border: 1px solid #2b313e;
          border-radius: 8px;
          overflow: hidden;
        }}
        #kg-wrapper:fullscreen {{
          width: 100vw;
          height: 100vh;
          border-radius: 0;
        }}
        #kg-network {{ width: 100%; height: 100%; }}
        .kg-toolbar {{
          position: absolute;
          top: 12px;
          right: 12px;
          z-index: 20;
          display: flex;
          align-items: center;
          gap: 8px;
          background: rgba(15, 23, 42, 0.88);
          border: 1px solid #334155;
          padding: 6px 10px;
          border-radius: 8px;
        }}
        .kg-btn {{
          background: #1e293b;
          color: #f8fafc;
          border: 1px solid #475569;
          border-radius: 6px;
          padding: 5px 10px;
          font-size: 12px;
          cursor: pointer;
        }}
        .kg-btn:hover {{ background: #334155; }}
        .kg-slider-wrap {{
          display: flex;
          align-items: center;
          gap: 6px;
          font-size: 12px;
        }}
        #pan-slider {{ width: 120px; cursor: pointer; }}
      </style>
    </head>
    <body>
      <div id="kg-wrapper">
        <div class="kg-toolbar">
          <div class="kg-slider-wrap">
            <span>? Pan ?</span>
            <input id="pan-slider" type="range" min="-800" max="800" value="0" step="10" />
          </div>
          <button class="kg-btn" id="fit-btn" title="Fit all nodes">Fit</button>
          <button class="kg-btn" id="fs-btn" title="Toggle Fullscreen">? Fullscreen</button>
        </div>
        <div id="kg-network"></div>
      </div>
      <script>
        const nodes = new vis.DataSet({nodes_json});
        const edges = new vis.DataSet({edges_json});
        const wrapper = document.getElementById("kg-wrapper");
        const container = document.getElementById("kg-network");
        const data = {{ nodes: nodes, edges: edges }};
        const options = {{
          nodes: {{
            shape: "dot",
            size: 18,
            font: {{ color: "#ffffff", size: 13 }},
            borderWidth: 2
          }},
          groups: {{
            Paper:   {{ color: {{ background: "#2563eb", border: "#93c5fd" }}, size: 22 }},
            Method:  {{ color: {{ background: "#16a34a", border: "#86efac" }}, size: 17 }},
            Dataset: {{ color: {{ background: "#ea580c", border: "#fdba74" }}, size: 17 }}
          }},
          edges: {{
            arrows: {{ to: {{ enabled: true, scaleFactor: 0.7 }} }},
            color: {{ color: "#64748b", highlight: "#38bdf8" }},
            font: {{ color: "#cbd5e1", size: 11, align: "middle", strokeWidth: 0 }},
            smooth: {{ type: "dynamic" }}
          }},
          physics: {{
            stabilization: true,
            barnesHut: {{ gravitationalConstant: -3500, springLength: 140 }}
          }},
          interaction: {{
            hover: true,
            tooltipDelay: 120,
            navigationButtons: true,
            keyboard: true
          }}
        }};
        const network = new vis.Network(container, data, options);

        // Horizontal Left/Right Slidebar
        const panSlider = document.getElementById("pan-slider");
        panSlider.addEventListener("input", () => {{
          const pos = network.getViewPosition();
          network.moveTo({{ position: {{ x: Number(panSlider.value), y: pos.y }}, animation: false }});
        }});

        // Fit to Screen Button
        document.getElementById("fit-btn").addEventListener("click", () => {{
          panSlider.value = 0;
          network.fit({{ animation: true }});
        }});

        // Fullscreen Toggle Button
        const fsBtn = document.getElementById("fs-btn");
        fsBtn.addEventListener("click", async () => {{
          if (!document.fullscreenElement) {{
            await wrapper.requestFullscreen();
          }} else {{
            await document.exitFullscreen();
          }}
        }});

        document.addEventListener("fullscreenchange", () => {{
          const isFs = Boolean(document.fullscreenElement);
          fsBtn.textContent = isFs ? "? Exit Fullscreen" : "? Fullscreen";
          setTimeout(() => {{
            network.redraw();
            network.fit({{ animation: true }});
          }}, 150);
        }});
      </script>
    </body>
    </html>
    """
    components.html(html_canvas, height=575)


# Sidebar: System Status, PDF Ingestion & Session Controls
with st.sidebar:
    st.header("System Telemetry")
    try:
        health = requests.get(f"{API_BASE_URL}/health", timeout=5).json()
        st.success("FastAPI Backend: Online")
        st.metric("FAISS Indexed Chunks", health["storage"]["faiss_vectors"])
        neo4j_info = health["storage"]["neo4j"]
        st.write(f"**Neo4j Graph:** {neo4j_info['status']} ({neo4j_info['papers']}P / {neo4j_info['methods']}M / {neo4j_info['datasets']}D)")
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

tab_research, tab_graph = st.tabs(["Autonomous Research", "Interactive Neo4j Knowledge Graph"])

with tab_research:
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

        st.subheader("Synthesized Literature Report")
        st.markdown(res["final_report"])

        st.download_button(
            label="Download Report (.md)",
            data=res["final_report"],
            file_name=f"research_report_{res['thread_id']}.md",
            mime="text/markdown",
        )

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

with tab_graph:
    st.subheader("Live Neo4j Entity & Relationship Explorer")
    st.caption("Use the ? Pan ? slidebar, navigation arrows, or ? Fullscreen button in the top-right corner of the graph.")
    if st.button("Refresh Graph from Neo4j"):
        st.rerun()
    try:
        kg_resp = requests.get(f"{API_BASE_URL}/graph", timeout=5).json()
        if kg_resp.get("status") == "offline":
            st.warning("Neo4j is currently offline. Start your local Neo4j instance to view the live graph.")
        else:
            render_interactive_kg(kg_resp)
    except Exception as exc:
        st.error(f"Could not fetch graph from FastAPI backend: {exc}")
