from pathlib import Path

from src.ingestion.pdf_processor import extract_text_from_pdf, extract_text_batch
from src.ingestion.chunker import chunk_text
from src.embeddings.embedder import Embedder
from src.retrieval.vector_store import VectorStore
from src.retrieval.retriever import Retriever
from src.generation.qa_pipeline import QAPipeline


class DocumentQAPipeline:
    def __init__(self):
        self.embedder = Embedder()
        self.vector_store = VectorStore(self.embedder)
        self.retriever = Retriever(self.vector_store)
        self.qa = None

    def ingest_pdf(self, pdf_path: str | Path) -> int:
        text = extract_text_from_pdf(pdf_path)
        source = Path(pdf_path).stem
        chunks = chunk_text(text, source=source)
        self.vector_store.add_documents(chunks)
        return len(chunks)

    def ingest_directory(self, pdf_dir: str | Path) -> int:
        docs = extract_text_batch(pdf_dir)
        total = 0
        for source, text in docs.items():
            chunks = chunk_text(text, source=source)
            self.vector_store.add_documents(chunks)
            total += len(chunks)
        return total

    def answer(self, query: str) -> dict:
        if self.qa is None:
            self.qa = QAPipeline()

        context = self.retriever.retrieve(query)
        if not context:
            return {"answer": "No documents found. Please ingest PDFs first.", "sources": []}

        prompt = self.qa.build_prompt(query, context)
        answer = self.qa.generate(prompt)

        return {
            "answer": answer,
            "sources": [
                {"source": c["source"], "relevance": f"{c['score']:.4f}"}
                for c in context
            ],
        }

    def document_count(self) -> int:
        return self.vector_store.count()

    def reset_index(self):
        self.vector_store.delete_all()
