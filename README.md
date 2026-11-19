# Research Intelligence Agent

[![CI - Test & Docker Build](https://github.com/rishi-ai-ds/research-intelligence-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/rishi-ai-ds/research-intelligence-agent/actions/workflows/ci.yml)
![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.42+-FF4B4B.svg)
![Neo4j](https://img.shields.io/badge/Neo4j-5.x-008CC1.svg)

An autonomous **Graph-RAG Literature Synthesis & Research Intelligence Agent** built with **LangGraph**, **FAISS**, **Neo4j**, **FastAPI**, and **Streamlit**. It decomposes complex research questions into targeted subtasks, retrieves and cross-verifies evidence across local vector/graph memory and live academic repositories (arXiv & PubMed), populates a persistent entity-relationship knowledge graph, and synthesizes grounded reports with traceable citations.

---

## System Architecture

```text
+-----------------------------------------------------------------------------+
|                        Streamlit Interactive UI (app.py)                    |
|  - Autonomous Research Workspace   - Live Neo4j Graph Explorer (Fullscreen) |
|  - PDF Knowledge Base Ingestion    - Citation & Execution Trace Inspector   |
+--------------------------------------+--------------------------------------+
                                       | HTTP / REST (JSON)
                                       v
+-----------------------------------------------------------------------------+
|                        FastAPI Backend (api/main.py)                        |
|   POST /research  |  POST /upload-pdf  |  GET /graph  |  GET /health        |
+--------------------------------------+--------------------------------------+
                                       |
                                       v
+-----------------------------------------------------------------------------+
|                  LangGraph Cyclic Agent Workflow (agent/)                   |
|                                                                             |
|   [Planner Node] ---> [Research & Retrieval Node] ---> [Evaluator Node]     |
|                              ^                                |             |
|                              |--- (Insufficient Evidence) ----+             |
|                                                               |             |
|                                                     (Sufficient / Max Loop) |
|                                                               v             |
|                                                     [Synthesizer Node]      |
+--------+-----------------------------+-----------------------------+--------+
         |                             |                             |
         v                             v                             v
+-----------------+          +-------------------+         +------------------+
|  FAISS Vector   |          |  Neo4j Knowledge  |         |  External APIs   |
|  Store (Local)  |          |  Graph (Bolt)     |         |  arXiv & PubMed  |
|  MiniLM-L6-v2   |          |  Paper / Method / |         |  & PyMuPDF PDFs  |
|                 |          |  Dataset Triples  |         |                  |
+-----------------+          +-------------------+         +------------------+

Key Features
Cyclic Agentic Workflow (LangGraph): Iterative Plan ? Retrieve ? Evaluate ? Synthesize loop with SQLite state checkpointing (memory/checkpointer.py) and configurable loop limits.

Hybrid Graph-RAG Memory:

FAISS Vector Store: Dense semantic chunk indexing using sentence-transformers/all-MiniLM-L6-v2 with automatic ingestion from research runs and uploaded PDFs.

Neo4j Knowledge Graph: Extracts and stores (Paper)-[:USES_METHOD]->(Method) and (Paper)-[:EVALUATED_ON]->(Dataset) triples with graceful offline fallback.

Interactive Neo4j Graph Explorer: Embedded vis-network canvas inside Streamlit featuring color-coded nodes, directed relationship arrows, horizontal pan slidebar, navigation controls, and fullscreen mode.

Resilient Multi-Provider LLM Routing: Configurable LLM_PROVIDER (groq, gemini, or auto) with automatic cross-provider failover on rate limits (429 RESOURCE_EXHAUSTED) and token/latency telemetry tracking.

Automated CI/CD: GitHub Actions pipeline running pytest unit/integration tests and Docker image build verification on every push to main.

Project Structure
Plaintext
research-intelligence-agent/
??? agent/                  # LangGraph state, nodes, conditional edges, and graph builder
??? api/                    # FastAPI server endpoints (/research, /upload-pdf, /graph, /health)
??? config/                 # Pydantic settings and environment configuration
??? evaluation/             # RAG grounding, citation accuracy, and telemetry benchmarks
??? memory/                 # FAISS vector store, Neo4j knowledge graph client, SQLite checkpointer
??? models/                 # Structured Pydantic schemas for plans, insights, and evaluations
??? tests/                  # Automated pytest suite
??? tools/                  # Multi-provider LLM client, arXiv/PubMed search, PyMuPDF extractor
??? app.py                  # Streamlit frontend with interactive Neo4j graph explorer
??? Dockerfile              # Container build definition
??? requirements.txt        # Python dependencies
Quickstart
1. Clone & Set Up Virtual Environment
Bash
git clone [https://github.com/rishi-ai-ds/research-intelligence-agent.git](https://github.com/rishi-ai-ds/research-intelligence-agent.git)
cd research-intelligence-agent
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
2. Configure Environment Variables
Create a .env file in the project root:

Code snippet
GEMINI_API_KEY=your_gemini_api_key
GROQ_API_KEY=your_groq_api_key
LLM_PROVIDER=groq
NEO4J_URI=bolt://127.0.0.1:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password
3. Run Backend & Frontend
Start the FastAPI server in Terminal 1:

Bash
uvicorn api.main:app --reload
Start the Streamlit UI in Terminal 2:

Bash
streamlit run app.py
Running Tests
Bash
python -m pytest -v
