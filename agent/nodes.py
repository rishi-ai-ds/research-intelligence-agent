from agent.state import AgentState
from models.schemas import ResearchPlan, PaperInsight, EvaluationResult
from tools.llm_client import generate_structured, generate_text
from tools.paper_search import arxiv_search, pubmed_search


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
    """Executes tool search for the active subtask and extracts structured PaperInsights."""
    iteration = state.get("iteration", 0) + 1
    idx = state.get("current_subtask_idx", 0)
    plan = state.get("plan", [])
    existing_evidence = state.get("evidence", [])

    if idx < len(plan):
        task = plan[idx]
        subtask_question = task["question"]
        query = task["search_query"]
        tool_name = task.get("recommended_tool", "arxiv_search")
    else:
        subtask_question = state.get("evaluation_feedback") or state["objective"]
        query = state["objective"]
        tool_name = "arxiv_search"

    # Execute real external search tool
    if tool_name == "pubmed_search":
        raw_papers = pubmed_search.invoke({"query": query, "max_results": 3})
    else:
        raw_papers = arxiv_search.invoke({"query": query, "max_results": 3})

    valid_papers = [p for p in raw_papers if "error" not in p]
    new_evidence = []
    citation_counter = len(existing_evidence) + 1

    for paper in valid_papers[:2]:
        # Extract structured insight using fast Groq model
        insight = generate_structured(
            schema=PaperInsight,
            system_prompt="Extract structured research metadata from this paper title and abstract. Be factual and concise.",
            user_prompt=f"Title: {paper['title']}\nAbstract: {paper['abstract']}",
            prefer_fast=True,
        )
        new_evidence.append(
            {
                "citation_id": f"[C{citation_counter}]",
                "paper_id": paper["paper_id"],
                "title": paper["title"],
                "url": paper.get("url", ""),
                "source_type": paper["source"],
                "subtask": subtask_question,
                "snippet": insight.key_findings or paper["abstract"][:300],
                "methods": insight.methods,
                "datasets": insight.datasets,
                "limitations": insight.limitations,
            }
        )
        citation_counter += 1

    next_idx = idx + 1
    return {
        "iteration": iteration,
        "current_subtask_idx": next_idx,
        "evidence": new_evidence,
        "logs": [
            f"[Research] Iter {iteration} ({tool_name} | query='{query}'): "
            f"extracted {len(new_evidence)} grounded citations."
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

    # Ensure every planned subtask has been attempted at least once
    if state.get("current_subtask_idx", 0) < len(plan):
        return {
            "is_sufficient": False,
            "evaluation_feedback": "Remaining planned subtasks still need execution.",
            "logs": [
                f"[Evaluator] Iter {iteration}: Subtask {state.get('current_subtask_idx')}/{len(plan)} "
                f"complete. Continuing to next subtask."
            ],
        }

    # Semantic sufficiency check via LLM
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
        f"{e['citation_id']} Title: {e['title']} ({e['paper_id']})\n"
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
        references.append(f"- **{e['citation_id']}** {e['title']} ({e['paper_id']}){url_part}")

    full_report = f"{body.strip()}\n\n" + "\n".join(references)
    return {
        "final_report": full_report,
        "logs": [f"[Report] Synthesized grounded report with {len(evidence)} citations."],
    }
