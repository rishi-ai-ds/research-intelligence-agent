import re
import math
import time
import json
from datasets import Dataset
from ragas import evaluate, RunConfig
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from config.settings import settings
from agent.graph import build_research_graph
from tools.llm_client import get_gemini_llm, usage_tracker
from memory.vector_store import get_embeddings


def compute_citation_grounding(report: str, evidence: list[dict]) -> dict:
    """Checks that every [C#] citation in the report exists in retrieved evidence."""
    valid_tags = {e["citation_id"] for e in evidence}
    used_tags = set(re.findall(r"\[C\d+\]", report))
    hallucinated_tags = used_tags - valid_tags
    coverage = len(used_tags & valid_tags) / len(valid_tags) if valid_tags else 0.0

    return {
        "valid_citations_available": len(valid_tags),
        "citations_used_in_report": len(used_tags),
        "hallucinated_citations": list(hallucinated_tags),
        "citation_coverage_ratio": round(coverage, 3),
    }


def _safe_score(val) -> float | None:
    try:
        f = float(val)
        return None if math.isnan(f) else round(f, 3)
    except Exception:
        return None


def run_evaluation(objective: str = "Compare transformer models for fake news detection") -> dict:
    """Runs the Research Intelligence Agent and evaluates output quality with Ragas."""
    graph = build_research_graph()

    start_time = time.perf_counter()
    result = graph.invoke({"objective": objective, "max_iterations": 2})
    elapsed_sec = round(time.perf_counter() - start_time, 2)

    report = result.get("final_report", "")
    # Strip References footer so Ragas evaluates factual prose claims cleanly
    prose_body = report.split("## References")[0].strip()[:1200]

    evidence = result.get("evidence", [])
    contexts = [
        f"{e['title']}: {e['snippet']} (Methods: {', '.join(e.get('methods', []))}; "
        f"Datasets: {', '.join(e.get('datasets', []))})"
        for e in evidence
    ] or ["No context retrieved."]

    grounding_stats = compute_citation_grounding(report, evidence)

    eval_dataset = Dataset.from_dict(
        {
            "question": [objective],
            "answer": [prose_body],
            "contexts": [contexts],
        }
    )

    evaluator_llm = LangchainLLMWrapper(get_gemini_llm(temperature=0.0))
    evaluator_embeddings = LangchainEmbeddingsWrapper(get_embeddings())

    ragas_scores = evaluate(
        dataset=eval_dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        run_config=RunConfig(max_workers=1, timeout=90, max_retries=2),
        raise_exceptions=False,
    )
    scores_dict = ragas_scores.to_pandas().iloc[0].to_dict()

    summary = {
        "objective": objective,
        "latency_seconds": elapsed_sec,
        "ragas_metrics": {
            "faithfulness": _safe_score(scores_dict.get("faithfulness")),
            "answer_relevancy": _safe_score(scores_dict.get("answer_relevancy")),
        },
        "grounding": grounding_stats,
        "llm_telemetry": usage_tracker.summary(),
    }

    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    out_file = settings.storage_dir / "eval_results.json"
    out_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    report_summary = run_evaluation()
    print(json.dumps(report_summary, indent=2))
