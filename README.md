# PDF Intelligence

A local RAG (Retrieval-Augmented Generation) system for PDF documents. Ask questions about your PDFs using fully local AI models.

## Features

- **Local PDF Processing**: Extract and chunk text from PDFs locally using PyMuPDF and LangChain.
- **Local Embeddings**: Generate vector representations of text using Sentence-Transformers (`all-MiniLM-L6-v2` by default).
- **Custom Vector Store**: Efficient local storage and retrieval using NumPy-based similarity search.
- **Local LLM Generation**: Generate answers using HuggingFace Transformers (`SmolLM2-1.7B-Instruct` by default).
- **Streamlit Web UI**: Easy-to-use interface for uploading PDFs, ingesting documents, and interactive Q&A.
- **CLI Ingestion**: Batch process directories of PDFs from the command line.

## Project Structure

```text
.
├── data/               # PDF storage and local vector index
├── src/
│   ├── ingestion/      # PDF processing and chunking
│   ├── embeddings/     # Embedding generation
│   ├── retrieval/      # Vector store and retrieval logic
│   ├── generation/     # LLM QA pipeline
│   ├── pipeline.py     # Main orchestration logic
│   └── config.py       # Configuration management
├── ui/
│   └── streamlit_app.py # Web interface
├── ingest.py           # CLI ingestion script
└── requirements.txt    # Python dependencies
```

## Setup

### Prerequisites

- Python 3.9+
- (Optional) CUDA-capable GPU for faster inference

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd pdf-intelligence
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scriptsctivate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure environment variables:
   ```bash
   cp .env.example .env
   ```
   Edit `.env` to change models or parameters if needed.

## Usage

### 1. Ingest Documents

You can ingest PDFs via the command line:

```bash
# Ingest from default directory (data/pdfs)
python ingest.py

# Ingest from a specific directory
python ingest.py /path/to/your/pdfs
```

### 2. Run the Web UI

Launch the Streamlit interface to upload files and ask questions using the helper script:

```bash
./run.sh
```

Or run it directly:

```bash
streamlit run ui/streamlit_app.py
```

Open your browser to `http://localhost:8501`.

## Configuration

The application can be configured via environment variables in the `.env` file:

| Variable | Description | Default |
|----------|-------------|---------|
| `EMBEDDING_MODEL` | HuggingFace model for embeddings | `all-MiniLM-L6-v2` |
| `LLM_MODEL` | HuggingFace model for generation | `HuggingFaceTB/SmolLM2-1.7B-Instruct` |
| `LLM_DEVICE` | Device for LLM inference (`cpu`, `cuda`, `mps`) | `cpu` |
| `CHUNK_SIZE` | Size of text chunks for processing | `1000` |
| `CHUNK_OVERLAP` | Overlap between chunks | `200` |
| `TOP_K` | Number of context chunks to retrieve | `4` |

## License

MIT