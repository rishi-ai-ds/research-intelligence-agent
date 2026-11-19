# Research Intelligence Agent

Autonomous Graph-RAG Research Intelligence Agent built with LangGraph, FAISS, Neo4j, FastAPI, and Streamlit.

## Architecture
- **Planner (Gemini):** Decomposes research objectives into structured subtasks.
- **Research Executor:** Queries Neo4j Knowledge Graph and local FAISS index first, falls back to arXiv/PubMed APIs, extracts structured Pydantic insights via Groq, and caches new papers.
- **Evaluator:** Checks evidence coverage across bounded iterations before synthesizing a citation-grounded report.
- **Memory & API:** Persistent SQLite checkpointing (storage/checkpoints.db), FastAPI backend (pi/main:app), and Streamlit UI (pp.py).

## Run Locally
1. Start backend (Terminal 1): uvicorn api.main:app --reload
2. Start frontend (Terminal 2): streamlit run app.py
3. Run tests: pytest -v
4. Run evaluation: python -m evaluation.ragas_eval

## Run with Docker
docker compose up --build
