# Streamlit in PDF Intelligence

This project uses **Streamlit** to provide a user-friendly interface for interacting with the local RAG (Retrieval-Augmented Generation) system.

## What is Streamlit?

Streamlit is an open-source Python library that makes it easy to create and share custom web apps for machine learning and data science. Instead of building a complex backend and frontend separately, Streamlit allows you to build the entire UI using only Python.

### Key Features Used in This Project

1.  **Reactive UI**: Every time a user interacts with a widget (like typing in the chat or uploading a file), Streamlit reruns the Python script from top to bottom. This ensures the UI is always in sync with the application state.
2.  **st.session_state**: Since the script reruns frequently, we use session_state to store information that needs to persist across runs, such as:
    *   The current chat history.
    *   The status of the PDF ingestion process.
3.  **Caching (st.cache_resource)**: Loading a 1.7 Billion parameter LLM (like SmolLM2) or an embedding model into memory takes time. We use Streamlit's caching to load these models once and keep them in memory, making subsequent questions nearly instantaneous.
4.  **Layouts**: We use st.sidebar for configuration settings and file uploads, and the main area for the interactive chat interface.
5.  **File Uploader**: st.file_uploader allows users to easily add new PDFs to the system without manually moving files into directories.

## How to Run the App

You can run the app using the provided helper script:

```bash
./run.sh
```

Or manually via:

```bash
streamlit run ui/streamlit_app.py
```

## Why Streamlit for RAG?

Streamlit is ideal for RAG applications because it allows for rapid prototyping of the user experience. We can quickly visualize which document chunks were retrieved and how the LLM is responding, all while keeping the entire codebase in a single language (Python).