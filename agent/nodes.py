from agent.state import AgentState


def planner_node(state: AgentState) -> dict:
    """Decomposes the user's research objective into structured subtasks."""
    objective = state["objective"]
    subtasks = [
        f"Survey deep learning architectures for: {objective}",
        f"Compare benchmark datasets and metrics for: {objective}",
        f"Identify limitations and open research gaps for: {objective}",
    ]
    return {
        "plan": subtasks,
        "current_subtask_idx": 0,
        "iteration": 0,
        "is_sufficient": False,
        "logs": [f"[Planner] Created {len(subtasks)} subtasks for objective."],
    }


def research_node(state: AgentState) -> dict:
    """Executes one research iteration for the current subtask."""
    iteration = state.get("iteration", 0) + 1
    idx = state.get("current_subtask_idx", 0)
    plan = state.get("plan", [])

    current_task = plan[idx] if idx < len(plan) else state["objective"]

    # Simulated evidence item (will be replaced by real FAISS/Neo4j/arXiv tools in Phase 3-5)
    finding = {
        "iteration": iteration,
        "subtask": current_task,
        "summary": f"Collected structured findings for '{current_task}' (pass {iteration}).",
        "source": f"seed_corpus_iter_{iteration}",
    }

    next_idx = min(idx + 1, len(plan))

    return {
        "iteration": iteration,
        "current_subtask_idx": next_idx,
        "evidence": [finding],
        "logs": [f"[Research] Iteration {iteration}: completed '{current_task}'."],
    }


def evaluator_node(state: AgentState) -> dict:
    """Checks whether all planned subtasks have sufficient evidence."""
    plan = state.get("plan", [])
    evidence = state.get("evidence", [])
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 3)

    all_subtasks_covered = len(evidence) >= len(plan)

    if all_subtasks_covered:
        return {
            "is_sufficient": True,
            "evaluation_feedback": "All planned subtasks covered with evidence.",
            "logs": [f"[Evaluator] Iteration {iteration}: SUFFICIENT ({len(evidence)}/{len(plan)} subtasks)."],
        }

    if iteration >= max_iterations:
        return {
            "is_sufficient": True,
            "evaluation_feedback": "Reached max_iterations safety limit; proceeding with collected evidence.",
            "logs": [f"[Evaluator] Iteration {iteration}: MAX_ITERATIONS reached ({max_iterations}). Stopping loop."],
        }

    return {
        "is_sufficient": False,
        "evaluation_feedback": f"Missing evidence for remaining subtasks ({len(evidence)}/{len(plan)} complete).",
        "logs": [f"[Evaluator] Iteration {iteration}: INSUFFICIENT ({len(evidence)}/{len(plan)}). Looping back."],
    }


def report_node(state: AgentState) -> dict:
    """Synthesizes collected evidence into a structured markdown report."""
    objective = state["objective"]
    evidence = state.get("evidence", [])

    sections = [f"# Research Intelligence Report\n\n**Objective:** {objective}\n\n## Findings"]
    for idx, item in enumerate(evidence, 1):
        sections.append(
            f"### {idx}. {item['subtask']}\n"
            f"- **Finding:** {item['summary']}\n"
            f"- **Source:** {item['source']}"
        )

    return {
        "final_report": "\n\n".join(sections),
        "logs": ["[Report] Final structured report generated."],
    }
