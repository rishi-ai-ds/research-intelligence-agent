import uuid
import shutil
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field

from config.settings import settings
from agent.graph import build_research_graph
from memory.checkpointer import get_sqlite_checkpointer
from memory.vector_store import get_vector_store, index_papers
from memory.knowledge_graph import kg_client
from tools.pdf_extractor import extract_pdf_document
from tools.llm_client import usage_tracker

app = FastAPI(
    title="Research Intelligence Agent API",
    description="Production LangGraph Graph-RAG research assistant with FAISS, Neo4j, and SQLite persistence.",
    version="1.0.0",
)

checkpointer = get_sqlite_checkpointer()
research_graph = build_research_graph(checkpointer=checkpointer)


class ResearchRequest(BaseModel):
    objective: str = Field(..., min_length=5, description="Research question or literature review objective")
    max_iterations: int = Field(default=3, ge=1, le=6)
    thread_id: str | None = Field(default=None, description="Optional thread ID to resume/track session")


class ResearchResponse(BaseModel):
    thread_id: str
    objective: str
    plan: list[dict]
    evidence: list[dict]
    logs: list[str]
    final_report: str
    telemetry: dict


@app.get("/health")
def health_check():
    store = get_vector_store()
    faiss_vectors = store.index.ntotal if store is not None else 0
    return {
        "status": "healthy",
        "models": {
            "primary_llm": settings.gemini_model,
            "fast_llm": settings.groq_model,
            "embeddings": settings.embedding_model,
        },
        "storage": {
            "faiss_vectors": faiss_vectors,
            "neo4j": kg_client.get_graph_stats(),
            "sqlite_db": str(settings.sqlite_db_path),
        },
        "llm_usage": usage_tracker.summary(),
    }


@app.post("/research", response_model=ResearchResponse)
def run_research(req: ResearchRequest):
    thread_id = req.thread_id or f"session-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}

    try:
        result = research_graph.invoke(
            {
                "objective": req.objective,
                "max_iterations": req.max_iterations,
            },
            config=config,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {exc}")

    return ResearchResponse(
        thread_id=thread_id,
        objective=req.objective,
        plan=result.get("plan", []),
        evidence=result.get("evidence", []),
        logs=result.get("logs", []),
        final_report=result.get("final_report", ""),
        telemetry=usage_tracker.summary(),
    )


@app.get("/threads/{thread_id}")
def get_thread_state(thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = research_graph.get_state(config)
    if not snapshot or not snapshot.values:
        raise HTTPException(status_code=404, detail=f"No saved state found for thread '{thread_id}'")
    return {
        "thread_id": thread_id,
        "values": snapshot.values,
        "next": snapshot.next,
    }


@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are supported.")

    settings.data_dir.mkdir(parents=True, exist_ok=True)
    dest_path = settings.data_dir / Path(file.filename).name

    with dest_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    doc_dict = extract_pdf_document(dest_path)
    chunks_added = index_papers([doc_dict])
    return {
        "filename": dest_path.name,
        "paper_id": doc_dict["paper_id"],
        "pages": doc_dict["page_count"],
        "chunks_indexed": chunks_added,
    }


@app.get("/graph")
def get_knowledge_graph(limit: int = 120):
    return kg_client.get_visual_graph(limit=limit)
