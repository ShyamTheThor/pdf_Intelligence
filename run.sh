


#!/bin/bash

# =================================================================
# PDF Intelligence - Streamlit Launcher
# =================================================================
# This script simplifies running the local RAG web interface.
# 
# What is Streamlit?
# 1. Framework: Turns Python scripts into interactive web apps.
# 2. Reactive: Reruns the script on every user interaction.
# 3. Widgets: Uses st.file_uploader, st.chat_input, etc.
# 4. Efficient: Uses @st.cache_resource to keep models in RAM.
# =================================================================

# 1. Activate Virtual Environment (if it exists)
if [ -d "venv" ]; then
    echo "--- Activating virtual environment ---"
    source venv/bin/activate
else
    echo "--- Note: 'venv' directory not found. Using system Python. ---"
fi

# 2. Check for Streamlit installation
if ! command -v streamlit &> /dev/null; then
    echo "Error: Streamlit is not installed. Run 'pip install -r requirements.txt' first."
    exit 1
fi

# 3. Run the App
echo "--- Starting Streamlit Web UI ---"
streamlit run ui/streamlit_app.py
