# Semantic Similarity Benchmark

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://semantic-similarity-benchmark-6qyh3kywcjy2x7urhu5hbp.streamlit.app/)

An interactive tool for analyzing and comparing text embedding models. This application visualizes how different architectures—static (GloVe) vs. context-aware (Transformers)—handle semantic similarity, sentiment analysis, and word order sensitivity.

## Key Features

* **Multi-Model Comparison**: Support for three distinct embedding architectures:
    * **GloVe** (Global Vectors): Static word-averaging (50d, 100d, etc.).
    * **Sentence Transformers**: Local context-aware models (e.g., `all-MiniLM-L6-v2`).
    * **OpenAI Embeddings**: SOTA API-based models (`text-embedding-3-small/large`).
* **Visual Analytics**: Real-time Pie Chart visualizations of cosine similarity distributions.
* **Semantic Search**: Compute similarity between input queries and dynamic categories.
* **Efficiency**: Implemented with caching mechanisms (`st.session_state`) to optimize performance.

## Tech Stack

* **Frontend**: Streamlit
* **Language**: Python 3.9+
* **Core Libraries**:
    * `numpy`: Vector arithmetic and cosine similarity calculations.
    * `sentence-transformers`: Hugging Face model integration.
    * `openai`: Integration with OpenAI API.
    * `gdown`: Automated data retrieval for GloVe embeddings.

## Installation & Setup

1.  **Clone the repository**
    ```bash
    git clone https://github.com/bohanc2/semantic-similarity-benchmark.git
    cd semantic-similarity-benchmark
    ```

2.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up Environment Variables**
    Create a `.env` file or export your OpenAI API key:
    ```bash
    export OPENAI_API_KEY="your-api-key-here"
    ```
    *(Note: For Streamlit Cloud deployment, set this in the "Secrets" management console)*

4.  **Run the Application**
    ```bash
    streamlit run miniproject_1_student.py
    ```

## Usage Guide

1.  **Select Model**: Choose between GloVe dimensions (50d/100d) on the sidebar.
2.  **Define Categories**: Input space-separated categories (e.g., `Food Animal Technology`).
3.  **Input Text**: Enter a sentence to analyze (e.g., *"I saw a Seattle dog hot"*).
4.  **Analyze**: View the cosine similarity distribution and compare how different models interpret the text.
