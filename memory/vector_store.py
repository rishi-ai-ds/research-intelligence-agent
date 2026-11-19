import json
from pathlib import Path
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from config.settings import settings
from tools.pdf_extractor import extract_pdf_document

_EMBEDDINGS: HuggingFaceEmbeddings | None = None
_VECTOR_STORE: FAISS | None = None


def get_embeddings() -> HuggingFaceEmbeddings:
    global _EMBEDDINGS
    if _EMBEDDINGS is None:
        _EMBEDDINGS = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _EMBEDDINGS


def get_vector_store() -> FAISS | None:
    """Loads the persisted FAISS index from disk if it exists."""
    global _VECTOR_STORE
    if _VECTOR_STORE is not None:
        return _VECTOR_STORE

    index_path = settings.faiss_index_dir
    if (index_path / "index.faiss").exists():
        _VECTOR_STORE = FAISS.load_local(
            str(index_path),
            get_embeddings(),
            allow_dangerous_deserialization=True,
        )
    return _VECTOR_STORE


def _save_vector_store(store: FAISS) -> None:
    global _VECTOR_STORE
    settings.faiss_index_dir.mkdir(parents=True, exist_ok=True)
    store.save_local(str(settings.faiss_index_dir))
    _VECTOR_STORE = store


def index_papers(papers: list[dict]) -> int:
    """Chunks, embeds, and upserts normalized paper dicts into the persistent FAISS index."""
    if not papers:
        return 0

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    docs: list[Document] = []

    for p in papers:
        paper_id = str(p.get("paper_id") or p.get("id") or "unknown")
        title = str(p.get("title", "Untitled")).strip()
        body = str(p.get("abstract") or p.get("summary") or p.get("text") or "").strip()
        if not body:
            continue

        full_content = f"Title: {title}\n\n{body}"
        chunks = splitter.split_text(full_content)
        for idx, chunk in enumerate(chunks):
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "paper_id": paper_id,
                        "title": title,
                        "source": p.get("source", "local_corpus"),
                        "url": p.get("url", ""),
                        "chunk_idx": idx,
                    },
                )
            )

    if not docs:
        return 0

    store = get_vector_store()
    if store is None:
        store = FAISS.from_documents(docs, get_embeddings())
    else:
        store.add_documents(docs)

    _save_vector_store(store)
    return len(docs)


def bootstrap_local_corpus() -> dict:
    """Indexes existing JSON datasets and PDFs from data/ into FAISS."""
    papers_to_index: list[dict] = []
    data_dir = settings.data_dir
    if not data_dir.exists():
        return {"papers_loaded": 0, "chunks_indexed": 0}

    # 1. Load JSON files (arxiv_papers.json, pubmed_multiple_queries.json, etc.)
    for json_file in data_dir.glob("*.json"):
        try:
            raw = json.loads(json_file.read_text(encoding="utf-8"))
            items = raw if isinstance(raw, list) else raw.get("papers", [])
            for i, item in enumerate(items):
                if isinstance(item, dict):
                    papers_to_index.append(
                        {
                            "paper_id": item.get("paper_id") or item.get("id") or f"{json_file.stem}:{i}",
                            "title": item.get("title", "Untitled"),
                            "abstract": item.get("abstract") or item.get("summary") or "",
                            "source": item.get("source") or json_file.stem,
                            "url": item.get("url", ""),
                        }
                    )
        except Exception:
            continue

    # 2. Load PDF files from data/
    for pdf_file in data_dir.glob("*.pdf"):
        try:
            doc_dict = extract_pdf_document(pdf_file)
            papers_to_index.append(doc_dict)
        except Exception:
            continue

    chunks_count = index_papers(papers_to_index)
    return {"papers_loaded": len(papers_to_index), "chunks_indexed": chunks_count}


@tool
def faiss_rag_search(query: str, top_k: int = 4) -> list[dict]:
    """Search the local FAISS vector database of indexed research papers and PDFs."""
    store = get_vector_store()
    if store is None:
        return []

    results = store.similarity_search_with_score(query, k=top_k)
    formatted = []
    for doc, distance in results:
        # Convert L2 distance on normalized vectors into a [0, 1] similarity score
        similarity = max(0.0, min(1.0, 1.0 - (float(distance) / 2.0)))
        formatted.append(
            {
                "paper_id": doc.metadata.get("paper_id", "local:doc"),
                "title": doc.metadata.get("title", "Indexed Paper"),
                "abstract": doc.page_content,
                "source": "faiss",
                "url": doc.metadata.get("url", ""),
                "relevance_score": round(similarity, 3),
            }
        )
    return formatted
