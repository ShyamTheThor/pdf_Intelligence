import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from src.config import PDF_DIR
from src.pipeline import DocumentQAPipeline


@st.cache_resource
def get_pipeline():
    return DocumentQAPipeline()


st.set_page_config(
    page_title="PDF Intelligence - Document QA",
    page_icon="📄",
    layout="wide",
)

st.title("PDF Intelligence")
st.markdown("Ask questions about your PDF documents using local AI.")

pipeline = get_pipeline()

tab1, tab2 = st.tabs(["Ask Questions", "Ingest Documents"])

with tab2:
    st.header("Document Ingestion")

    col1, col2 = st.columns(2)
    with col1:
        uploaded_files = st.file_uploader(
            "Upload PDFs",
            type="pdf",
            accept_multiple_files=True,
        )
        if uploaded_files:
            for uploaded_file in uploaded_files:
                save_path = PDF_DIR / uploaded_file.name
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            st.success(f"Saved {len(uploaded_files)} PDF(s) to {PDF_DIR}")

    with col2:
        st.subheader("Actions")
        if st.button("Ingest All PDFs", type="primary"):
            with st.spinner("Processing PDFs..."):
                count = pipeline.ingest_directory(PDF_DIR)
            st.success(f"Ingested {count} chunks from PDFs in {PDF_DIR}")

        if st.button("View Index Stats"):
            st.info(f"Total chunks in index: {pipeline.document_count()}")

        if st.button("Reset Index", type="secondary"):
            pipeline.reset_index()
            st.warning("Index cleared.")

with tab1:
    st.header("Question Answering")

    if pipeline.document_count() == 0:
        st.warning("No documents ingested yet. Go to 'Ingest Documents' tab first.")
    else:
        query = st.text_input("Ask a question about your documents:")
        if query:
            with st.spinner("Retrieving context and generating answer..."):
                result = pipeline.answer(query)

            st.subheader("Answer")
            st.write(result["answer"])

            with st.expander("View Sources"):
                for src in result["sources"]:
                    st.markdown(f"- **{src['source']}** (score: {src['relevance']})")

st.divider()
st.caption("Powered by HuggingFace Transformers + Sentence-Transformers + NumPy")
