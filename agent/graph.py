from langgraph.graph import StateGraph, START, END
from agent.state import AgentState
from agent.nodes import planner_node, research_node, evaluator_node, report_node
from agent.edges import route_after_evaluation


def build_research_graph(checkpointer=None):
    """Builds and compiles the Research Intelligence LangGraph workflow."""
    workflow = StateGraph(AgentState)

    # Register nodes
    workflow.add_node("planner", planner_node)
    workflow.add_node("research", research_node)
    workflow.add_node("evaluator", evaluator_node)
    workflow.add_node("report", report_node)

    # Define deterministic edges
    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "research")
    workflow.add_edge("research", "evaluator")

    # Define conditional loop edge
    workflow.add_conditional_edges(
        "evaluator",
        route_after_evaluation,
        {
            "research": "research",
            "report": "report",
        },
    )

    workflow.add_edge("report", END)

    return workflow.compile(checkpointer=checkpointer)
