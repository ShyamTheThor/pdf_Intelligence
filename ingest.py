#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import PDF_DIR
from src.pipeline import DocumentQAPipeline


def main():
    pdf_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else PDF_DIR

    if not pdf_dir.exists():
        print(f"Directory not found: {pdf_dir}")
        sys.exit(1)

    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {pdf_dir}")
        sys.exit(1)

    pipeline = DocumentQAPipeline()
    total = pipeline.ingest_directory(pdf_dir)

    print(f"\nDone! {total} total chunks stored ({pipeline.document_count()} in index).")


if __name__ == "__main__":
    main()
