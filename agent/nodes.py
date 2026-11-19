from agent.state import AgentState
from models.schemas import ResearchPlan, PaperInsight, EvaluationResult
from tools.llm_client import generate_structured, generate_text
from tools.paper_search import arxiv_search, pubmed_search
from memory.vector_store import faiss_rag_search, index_papers


def planner_node(state: AgentState) -> dict:
    """Uses Gemini structured output to decompose the objective into research subtasks."""
    objective = state["objective"]

    system_prompt = (
        "You are a senior AI research lead. Decompose the user's research objective into "
        "2 to 4 focused, non-overlapping subtasks. Choose 'pubmed_search' for biomedical/clinical "
        "topics and 'arxiv_search' for AI/ML/computer science topics. Keep search_query concise (3-6 keywords)."
    )
    plan_obj = generate_structured(
        schema=ResearchPlan,
        system_prompt=system_prompt,
        user_prompt=f"Research objective: {objective}",
        prefer_fast=False,
    )

    subtasks = [s.model_dump() for s in plan_obj.subtasks]
    return {
        "plan": subtasks,
        "current_subtask_idx": 0,
        "iteration": 0,
        "is_sufficient": False,
        "evaluation_feedback": "",
        "logs": [f"[Planner] Generated {len(subtasks)} structured subtasks."],
    }


def research_node(state: AgentState) -> dict:
    """Hybrid RAG retrieval: searches local FAISS first, falls back to arXiv/PubMed, and caches new papers."""
    iteration = state.get("iteration", 0) + 1
    idx = state.get("current_subtask_idx", 0)
    plan = state.get("plan", [])
    existing_evidence = state.get("evidence", [])
    seen_ids = {e["paper_id"] for e in existing_evidence}

    if idx < len(plan):
        task = plan[idx]
        subtask_question = task["question"]
        query = task["search_query"]
        external_tool = task.get("recommended_tool", "arxiv_search")
    else:
        subtask_question = state.get("evaluation_feedback") or state["objective"]
        query = state.get("evaluation_feedback") or state["objective"]
        external_tool = "arxiv_search"

    # Step 1: Query local FAISS index first
    local_hits = faiss_rag_search.invoke({"query": query, "top_k": 4})
    strong_local = [
        h for h in local_hits
        if h.get("relevance_score", 0) >= 0.58 and h["paper_id"] not in seen_ids
    ]

    candidate_papers = []
    source_used = "faiss_rag"

    if len(strong_local) >= 2:
        candidate_papers = strong_local[:2]
    else:
        # Step 2: Fetch fresh literature from arXiv or PubMed and cache into FAISS
        source_used = external_tool
        if external_tool == "pubmed_search":
            raw_external = pubmed_search.invoke({"query": query, "max_results": 4})
        else:
            raw_external = arxiv_search.invoke({"query": query, "max_results": 4})

        valid_external = [
            p for p in raw_external
            if "error" not in p and p["paper_id"] not in seen_ids
        ]
        if valid_external:
            index_papers(valid_external)  # Auto-expand local FAISS knowledge base

        # Combine any strong local hit with fresh external hits
        candidate_papers = (strong_local + valid_external)[:2]

    new_evidence = []
    citation_counter = len(existing_evidence) + 1

    for paper in candidate_papers:
        if paper["paper_id"] in seen_ids:
            continue
        seen_ids.add(paper["paper_id"])

        insight = generate_structured(
            schema=PaperInsight,
            system_prompt="Extract structured research metadata from this paper title and excerpt. Be factual and concise.",
            user_prompt=f"Title: {paper['title']}\nText: {paper['abstract']}",
            prefer_fast=True,
        )
        new_evidence.append(
            {
                "citation_id": f"[C{citation_counter}]",
                "paper_id": paper["paper_id"],
                "title": paper["title"],
                "url": paper.get("url", ""),
                "source_type": paper.get("source", "faiss"),
                "subtask": subtask_question,
                "snippet": insight.key_findings or paper["abstract"][:300],
                "methods": insight.methods,
                "datasets": insight.datasets,
                "limitations": insight.limitations,
                "relevance_score": paper.get("relevance_score", 0.75),
            }
        )
        citation_counter += 1

    next_idx = idx + 1
    return {
        "iteration": iteration,
        "current_subtask_idx": next_idx,
        "evidence": new_evidence,
        "logs": [
            f"[Research] Iter {iteration} ({source_used} | query='{query}'): "
            f"added {len(new_evidence)} unique citations (total={len(existing_evidence) + len(new_evidence)})."
        ],
    }


def evaluator_node(state: AgentState) -> dict:
    """Evaluates whether collected evidence covers the plan or requires another loop."""
    plan = state.get("plan", [])
    evidence = state.get("evidence", [])
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 3)

    if iteration >= max_iterations:
        return {
            "is_sufficient": True,
            "evaluation_feedback": f"Reached max_iterations ({max_iterations}); synthesizing report.",
            "logs": [f"[Evaluator] Iter {iteration}: MAX_ITERATIONS reached ({max_iterations})."],
        }

    if state.get("current_subtask_idx", 0) < len(plan):
        return {
            "is_sufficient": False,
            "evaluation_feedback": "Remaining planned subtasks still need execution.",
            "logs": [
                f"[Evaluator] Iter {iteration}: Subtask {state.get('current_subtask_idx')}/{len(plan)} "
                f"complete. Continuing to next subtask."
            ],
        }

    evidence_digest = "\n".join(
        f"{e['citation_id']} {e['title']} | Methods: {e['methods']} | Datasets: {e['datasets']}"
        for e in evidence
    )
    verdict = generate_structured(
        schema=EvaluationResult,
        system_prompt=(
            "You are a strict research quality reviewer. Determine if the collected citations "
            "adequately cover the user's objective. Set is_sufficient=True if key methods, datasets, "
            "and findings are present."
        ),
        user_prompt=f"Objective: {state['objective']}\n\nCollected Evidence:\n{evidence_digest}",
        prefer_fast=True,
    )

    return {
        "is_sufficient": verdict.is_sufficient,
        "evaluation_feedback": verdict.refined_query or verdict.feedback,
        "logs": [
            f"[Evaluator] Iter {iteration}: is_sufficient={verdict.is_sufficient} "
            f"(confidence={verdict.confidence_score:.2f})."
        ],
    }


def report_node(state: AgentState) -> dict:
    """Synthesizes a grounded, citation-backed Markdown research report."""
    objective = state["objective"]
    evidence = state.get("evidence", [])

    if not evidence:
        return {
            "final_report": f"# Research Report\n\nNo verifiable papers were retrieved for: **{objective}**.",
            "logs": ["[Report] Finished with empty evidence warning."],
        }

    evidence_context = "\n\n".join(
        f"{e['citation_id']} Title: {e['title']} ({e['paper_id']}) [via {e['source_type']}]\n"
        f"- Subtask: {e['subtask']}\n"
        f"- Methods: {', '.join(e['methods']) or 'Not specified'}\n"
        f"- Datasets: {', '.join(e['datasets']) or 'Not specified'}\n"
        f"- Key Finding: {e['snippet']}\n"
        f"- Limitations: {e['limitations'] or 'Not specified'}"
        for e in evidence
    )

    system_prompt = (
        "You are an expert AI Research Analyst. Write a clear, structured Markdown literature synthesis "
        "answering the user's objective using ONLY the provided evidence items. "
        "CRITICAL GROUNDING RULE: Cite claims inline using exact tags like [C1], [C2]. "
        "Never invent papers, metrics, or citations not present in the evidence."
    )
    body = generate_text(
        system_prompt=system_prompt,
        user_prompt=f"Objective: {objective}\n\nVerified Evidence:\n{evidence_context}",
        temperature=0.2,
    )

    references = ["## References"]
    for e in evidence:
        url_part = f" — {e['url']}" if e.get("url") else ""
        references.append(f"- **{e['citation_id']}** {e['title']} ({e['paper_id']}) [{e['source_type']}]{url_part}")

    full_report = f"{body.strip()}\n\n" + "\n".join(references)
    return {
        "final_report": full_report,
        "logs": [f"[Report] Synthesized grounded report with {len(evidence)} unique citations."],
    }
