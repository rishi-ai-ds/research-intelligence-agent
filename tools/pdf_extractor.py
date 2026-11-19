from pathlib import Path
import pymupdf


def extract_pdf_document(pdf_path: str | Path) -> dict:
    """Extracts clean page-by-page text and metadata from a research paper PDF."""
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    doc = pymupdf.open(str(path))
    pages = []
    for page_num in range(len(doc)):
        text = doc[page_num].get_text("text").strip()
        if text:
            pages.append({"page": page_num + 1, "text": text})

    full_text = "\n\n".join(p["text"] for p in pages)
    metadata = doc.metadata or {}
    doc.close()

    return {
        "paper_id": f"pdf:{path.stem}",
        "title": metadata.get("title") or path.stem.replace("_", " "),
        "page_count": len(pages),
        "text": full_text,
        "pages": pages,
        "source": "pdf",
    }
