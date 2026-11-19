from typing import Literal
from pydantic import BaseModel, Field


class Subtask(BaseModel):
    step_id: int = Field(description="Sequential step number starting at 1")
    question: str = Field(description="Specific sub-question or research goal")
    search_query: str = Field(description="Concise keyword query for vector/paper search")
    recommended_tool: Literal["faiss_rag", "neo4j_kg", "arxiv_search", "pubmed_search"] = Field(
        description="Best tool to answer this subtask"
    )


class ResearchPlan(BaseModel):
    objective_summary: str = Field(description="Refined understanding of the user's research objective")
    subtasks: list[Subtask] = Field(min_length=2, max_length=5, description="2 to 5 ordered research subtasks")


class PaperInsight(BaseModel):
    domain: list[str] = Field(default_factory=list)
    research_problem: str = ""
    methods: list[str] = Field(default_factory=list)
    datasets: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    key_findings: str = ""
    limitations: str = ""
    future_directions: str = ""


class EvidenceItem(BaseModel):
    citation_id: str = Field(description="Unique citation tag such as [C1], [C2]")
    paper_id: str = Field(description="arXiv ID, PMID, or document UUID")
    title: str = Field(description="Exact paper title")
    source_type: Literal["faiss", "neo4j", "arxiv", "pubmed", "pdf"]
    subtask: str = Field(description="Subtask this evidence supports")
    snippet: str = Field(description="Extracted relevant text or structured finding")
    methods: list[str] = Field(default_factory=list)
    datasets: list[str] = Field(default_factory=list)
    limitations: str = ""
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)


class EvaluationResult(BaseModel):
    is_sufficient: bool = Field(description="True if collected evidence thoroughly answers all subtasks")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Confidence score from 0.0 to 1.0")
    feedback: str = Field(description="Explanation of what is covered and what is missing")
    refined_query: str = Field(
        default="",
        description="Targeted follow-up search query if is_sufficient is False",
    )
