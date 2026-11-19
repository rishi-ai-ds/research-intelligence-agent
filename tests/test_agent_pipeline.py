from fastapi.testclient import TestClient
from agent.edges import route_after_evaluation
from models.schemas import ResearchPlan, Subtask, PaperInsight
from memory.vector_store import faiss_rag_search
from api.main import app


def test_conditional_routing_continues_when_insufficient():
    state = {"is_sufficient": False, "iteration": 1, "max_iterations": 3}
    assert route_after_evaluation(state) == "research"


def test_conditional_routing_stops_on_sufficiency():
    state = {"is_sufficient": True, "iteration": 1, "max_iterations": 3}
    assert route_after_evaluation(state) == "report"


def test_conditional_routing_stops_on_max_iterations():
    state = {"is_sufficient": False, "iteration": 3, "max_iterations": 3}
    assert route_after_evaluation(state) == "report"


def test_pydantic_schemas_validation():
    plan = ResearchPlan(
        objective_summary="Evaluate GNNs for fake news",
        subtasks=[
            Subtask(step_id=1, question="Q1", search_query="GNN fake news", recommended_tool="faiss_rag"),
            Subtask(step_id=2, question="Q2", search_query="fake news datasets", recommended_tool="arxiv_search"),
        ],
    )
    assert len(plan.subtasks) == 2

    insight = PaperInsight(methods=["GraphSAGE"], datasets=["FakeNewsNet"], key_findings="Improved F1 by 4%.")
    assert "GraphSAGE" in insight.methods


def test_faiss_retrieval_returns_scored_results():
    hits = faiss_rag_search.invoke({"query": "fake news detection machine learning", "top_k": 2})
    assert isinstance(hits, list)
    assert len(hits) >= 1
    assert 0.0 <= hits[0]["relevance_score"] <= 1.0


def test_fastapi_health_endpoint():
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "healthy"
    assert "faiss_vectors" in body["storage"]
