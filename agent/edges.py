from typing import Literal
from agent.state import AgentState


def route_after_evaluation(state: AgentState) -> Literal["research", "report"]:
    """Decides whether to continue researching or generate the final report."""
    if state.get("is_sufficient", False):
        return "report"

    if state.get("iteration", 0) >= state.get("max_iterations", 3):
        return "report"

    return "research"
