import operator
from typing import Annotated, TypedDict


class AgentState(TypedDict):
    # Input
    objective: str

    # Planning
    plan: list[str]
    current_subtask_idx: int

    # Iterative Research & Accumulation (operator.add appends across loops)
    evidence: Annotated[list[dict], operator.add]
    logs: Annotated[list[str], operator.add]

    # Evaluation & Loop Control
    iteration: int
    max_iterations: int
    is_sufficient: bool
    evaluation_feedback: str

    # Output
    final_report: str
