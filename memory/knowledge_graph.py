import time
import logging
from neo4j import GraphDatabase
from langchain_core.tools import tool
from config.settings import settings

logger = logging.getLogger(__name__)


class KnowledgeGraphClient:
    """Manages Neo4j connection, schema constraints, triple ingestion, and graph queries."""

    def __init__(self):
        self._driver = None
        self._last_fail_ts = 0.0

    def _get_driver(self):
        if self._driver is not None:
            return self._driver
        if time.time() - self._last_fail_ts < 30.0:
            return None

        # Use direct bolt:// for local single-instance Neo4j to avoid routing table errors
        uri = settings.neo4j_uri
        if "127.0.0.1" in uri or "localhost" in uri:
            uri = uri.replace("neo4j://", "bolt://")

        try:
            driver = GraphDatabase.driver(
                uri,
                auth=(settings.neo4j_user, settings.neo4j_password),
                connection_timeout=3.0,
            )
            driver.verify_connectivity()
            self._driver = driver
            self._ensure_constraints()
            return self._driver
        except Exception as exc:
            self._last_fail_ts = time.time()
            logger.warning(f"Neo4j offline ({exc}); continuing with FAISS + external tools.")
            return None

    def _ensure_constraints(self) -> None:
        if not self._driver:
            return
        queries = [
            "CREATE CONSTRAINT paper_id_unique IF NOT EXISTS FOR (p:Paper) REQUIRE p.paper_id IS UNIQUE",
            "CREATE CONSTRAINT method_name_unique IF NOT EXISTS FOR (m:Method) REQUIRE m.name IS UNIQUE",
            "CREATE CONSTRAINT dataset_name_unique IF NOT EXISTS FOR (d:Dataset) REQUIRE d.name IS UNIQUE",
        ]
        with self._driver.session() as session:
            for q in queries:
                session.run(q)

    def upsert_paper_triples(
        self,
        paper_id: str,
        title: str,
        source: str,
        methods: list[str],
        datasets: list[str],
        url: str = "",
    ) -> bool:
        """Upserts Paper -> USES_METHOD -> Method and Paper -> EVALUATED_ON -> Dataset triples."""
        driver = self._get_driver()
        if not driver:
            return False

        clean_methods = [m.strip() for m in methods if m and m.strip()]
        clean_datasets = [d.strip() for d in datasets if d and d.strip()]

        cypher = """
        MERGE (p:Paper {paper_id: })
        SET p.title = , p.source = , p.url = 
        WITH p
        UNWIND CASE WHEN size() = 0 THEN [null] ELSE  END AS m_name
        FOREACH (_ IN CASE WHEN m_name IS NOT NULL THEN [1] ELSE [] END |
            MERGE (m:Method {name: m_name})
            MERGE (p)-[:USES_METHOD]->(m)
        )
        WITH p
        UNWIND CASE WHEN size() = 0 THEN [null] ELSE  END AS d_name
        FOREACH (_ IN CASE WHEN d_name IS NOT NULL THEN [1] ELSE [] END |
            MERGE (d:Dataset {name: d_name})
            MERGE (p)-[:EVALUATED_ON]->(d)
        )
        """
        with driver.session() as session:
            session.run(
                cypher,
                paper_id=paper_id,
                title=title,
                source=source,
                url=url,
                methods=clean_methods,
                datasets=clean_datasets,
            )
        return True

    def search_entities(self, keyword: str, limit: int = 5) -> list[dict]:
        """Finds papers linked to matching Methods, Datasets, or Titles in Neo4j."""
        driver = self._get_driver()
        if not driver:
            return []

        cypher = """
        MATCH (p:Paper)
        OPTIONAL MATCH (p)-[:USES_METHOD]->(m:Method)
        OPTIONAL MATCH (p)-[:EVALUATED_ON]->(d:Dataset)
        WITH p, collect(DISTINCT m.name) AS methods, collect(DISTINCT d.name) AS datasets
        WHERE toLower(p.title) CONTAINS toLower()
           OR any(x IN methods WHERE toLower(x) CONTAINS toLower())
           OR any(y IN datasets WHERE toLower(y) CONTAINS toLower())
        RETURN p.paper_id AS paper_id,
               p.title AS title,
               p.url AS url,
               methods,
               datasets
        LIMIT 
        """
        with driver.session() as session:
            records = session.run(cypher, kw=keyword, limit=limit)
            results = []
            for rec in records:
                methods_str = ", ".join(rec["methods"]) or "None"
                datasets_str = ", ".join(rec["datasets"]) or "None"
                results.append(
                    {
                        "paper_id": rec["paper_id"],
                        "title": rec["title"],
                        "url": rec["url"] or "",
                        "source": "neo4j",
                        "abstract": f"Knowledge Graph links — Methods: [{methods_str}] | Datasets: [{datasets_str}]",
                        "methods": rec["methods"],
                        "datasets": rec["datasets"],
                        "relevance_score": 0.85,
                    }
                )
            return results

    def get_graph_stats(self) -> dict:
        """Returns node counts (or offline status if Neo4j is not running)."""
        driver = self._get_driver()
        if not driver:
            return {"status": "offline", "papers": 0, "methods": 0, "datasets": 0}

        cypher = """
        OPTIONAL MATCH (p:Paper) WITH count(p) AS papers
        OPTIONAL MATCH (m:Method) WITH papers, count(m) AS methods
        OPTIONAL MATCH (d:Dataset) RETURN papers, methods, count(d) AS datasets
        """
        with driver.session() as session:
            rec = session.run(cypher).single()
            return {
                "status": "connected",
                "papers": rec["papers"],
                "methods": rec["methods"],
                "datasets": rec["datasets"],
            }


kg_client = KnowledgeGraphClient()


@tool
def neo4j_kg_search(entity_query: str, limit: int = 5) -> list[dict]:
    """Query the Neo4j Research Knowledge Graph for papers by Method, Dataset, or Title."""
    return kg_client.search_entities(keyword=entity_query, limit=limit)
