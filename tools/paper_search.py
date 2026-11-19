import urllib.parse
import requests
import feedparser
from langchain_core.tools import tool


def fetch_arxiv_raw(query: str, max_results: int = 5) -> list[dict]:
    """Queries the arXiv Atom API and returns normalized paper metadata."""
    encoded_query = urllib.parse.quote(f"all:{query}")
    url = (
        f"http://export.arxiv.org/api/query?"
        f"search_query={encoded_query}&start=0&max_results={max_results}"
        f"&sortBy=relevance&sortOrder=descending"
    )
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    feed = feedparser.parse(resp.text)

    papers = []
    for entry in feed.entries:
        raw_id = entry.get("id", "").split("/abs/")[-1]
        authors = [a.name for a in entry.get("authors", [])]
        papers.append(
            {
                "paper_id": f"arxiv:{raw_id}",
                "title": " ".join(entry.get("title", "").split()),
                "abstract": " ".join(entry.get("summary", "").split()),
                "authors": authors[:5],
                "published": entry.get("published", "")[:10],
                "source": "arxiv",
                "url": entry.get("link", ""),
            }
        )
    return papers


def fetch_pubmed_raw(query: str, max_results: int = 5) -> list[dict]:
    """Queries NCBI PubMed E-utilities (esearch + esummary) and returns normalized metadata."""
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    search_params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
        "sort": "relevance",
    }
    search_resp = requests.get(search_url, params=search_params, timeout=15)
    search_resp.raise_for_status()
    pmids = search_resp.json().get("esearchresult", {}).get("idlist", [])
    if not pmids:
        return []

    summary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    summary_params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "json",
    }
    sum_resp = requests.get(summary_url, params=summary_params, timeout=15)
    sum_resp.raise_for_status()
    result_map = sum_resp.json().get("result", {})

    papers = []
    for pmid in pmids:
        item = result_map.get(pmid)
        if not item:
            continue
        authors = [a.get("name", "") for a in item.get("authors", []) if a.get("name")]
        papers.append(
            {
                "paper_id": f"pubmed:{pmid}",
                "title": item.get("title", "").strip(),
                "abstract": f"PubMed article published in {item.get('source', 'journal')} ({item.get('pubdate', '')}).",
                "authors": authors[:5],
                "published": item.get("pubdate", ""),
                "source": "pubmed",
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            }
        )
    return papers


@tool
def arxiv_search(query: str, max_results: int = 4) -> list[dict]:
    """Search arXiv for computer science, AI, and machine learning research papers."""
    try:
        return fetch_arxiv_raw(query=query, max_results=max_results)
    except Exception as exc:
        return [{"error": f"arXiv search failed: {exc}"}]


@tool
def pubmed_search(query: str, max_results: int = 4) -> list[dict]:
    """Search PubMed for biomedical, clinical, and healthcare research literature."""
    try:
        return fetch_pubmed_raw(query=query, max_results=max_results)
    except Exception as exc:
        return [{"error": f"PubMed search failed: {exc}"}]
