import fitz
from pathlib import Path


def extract_text_from_pdf(pdf_path: str | Path) -> str:
    doc = fitz.open(str(pdf_path))
    text_parts = []
    for page in doc:
        text_parts.append(page.get_text())
    doc.close()
    return "\n".join(text_parts)


def extract_text_batch(pdf_dir: str | Path) -> dict[str, str]:
    pdf_dir = Path(pdf_dir)
    results = {}
    for pdf_file in sorted(pdf_dir.glob("*.pdf")):
        results[pdf_file.stem] = extract_text_from_pdf(pdf_file)
    return results
