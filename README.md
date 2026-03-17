# EEP 596A — LLM Mini Project Portfolio

This repository contains my work for three mini projects in EEP 596A, focused on practical applications of large language models (LLMs), embeddings, tools, and multi‑agent systems.

- **Project 1**: Word embeddings and semantic search demo
- **Project 2**: Safety‑aware multi‑agent chatbot with retrieval and evaluation
- **Project 3**: FinTech Q&A chat app with single‑agent vs multi‑agent architectures

---

### Project 1 — Word Embeddings & Semantic Search (`MP1_P1`, `MP1_P2`)

**Goal**

Explore different text embedding methods and cosine similarity for semantic search, and build an interactive demo to visualize how embeddings behave on intuitive examples.

**Highlights**

- Implemented **cosine similarity**–based scoring for text similarity.
- Compared multiple embedding families:
  - **GloVe** word embeddings (25d / 50d / 100d, averaged over tokens).
  - **Sentence Transformers** (e.g., `all-MiniLM-L6-v2`).
  - **OpenAI embeddings** (`text-embedding-3-small` and `text-embedding-3-large`).
- Built a **Streamlit search demo** where the user:
  - Defines a list of categories (e.g., “Flowers Colors Cars Weather Food”).
  - Types a sentence.
  - Sees which category is closest under each embedding model (pie charts & tables).
- Added basic robustness:
  - Handles zero vectors and NaNs in embeddings.
  - Caches models and category embeddings to avoid recomputation.

**Key skills**

- Working with high‑dimensional embeddings and cosine similarity.
- Practical experience with OpenAI’s embedding API and Sentence Transformers.
- Building a small interactive UI with Streamlit to make model behavior observable.

---

### Project 2 — Multi‑Agent Chatbot & LLM‑as‑a‑Judge Evaluation (`MP2_P1P2`, `MP2_P3P4`)

**Goal**

Design and implement a safety‑aware multi‑agent chatbot that uses retrieval, a controller (“head”) agent, and a separate LLM‑based evaluation pipeline (“LLM‑as‑a‑Judge”).

**Architecture**

The system is built around several specialized agents:

- **Obnoxious Agent**
  - Filters user inputs that are rude, offensive, or inappropriate.
  - Returns a strict binary signal (OBNOXIOUS / NOT OBNOXIOUS).
- **Query / Retriever Agent**
  - Uses OpenAI embeddings and **Pinecone** as a vector store.
  - Converts user queries into embeddings, queries Pinecone, and returns top‑k matched documents plus metadata.
- **Relevant Documents Agent**
  - Judges whether retrieved documents are actually relevant to the current query.
  - Outputs a simple decision (Relevant / Irrelevant) to gate downstream answering.
- **Answering Agent**
  - Generates answers only when relevant documents exist.
  - Conditions its response on both retrieved context and conversation history.
- **Head Agent (Controller)**
  - Orchestrates all sub‑agents in a loop:
    1. Check if the query is obnoxious; if yes, refuse.
    2. Retrieve documents from Pinecone.
    3. Check relevance of documents.
    4. Either refuse as “out of scope” or answer using the context.
  - Maintains conversation history across turns.

**Part 4 — LLM‑as‑a‑Judge Evaluation**

- Implemented a **test dataset generator** that can synthesize prompts for categories like:
  - Obnoxious queries
  - Irrelevant queries
  - Relevant queries
  - Greetings / small talk
  - Hybrid prompts (mixed relevant + irrelevant/obnoxious content)
  - Multi‑turn conversations
- Implemented an **LLM Judge** that:
  - Takes user input, chatbot response, and the agent path.
  - Outputs a binary score (1 = desired behavior, 0 = undesired).
- Built an **evaluation pipeline** that:
  - Runs the full test set through the chatbot.
  - Aggregates scores per category and reports overall accuracy.
  - Helps identify weaknesses such as over‑refusal or under‑refusal.

**Key skills**

- Designing multi‑agent LLM systems with explicit safety and relevance stages.
- Integrating OpenAI, Pinecone, and simple evaluation tooling.
- Experimenting with “LLM‑as‑a‑Judge” to systematically assess agent behavior.

---

### Project 3 — FinTech Q&A Chat with Single vs Multi‑Agent Architectures (`MP3`)

**Goal**

Build a finance‑focused chat application that compares a single tool‑using agent to a pipeline of specialized agents, all exposed via a Streamlit chat interface.

**Data & Tools**

- **Market data & prices** via `yfinance`.
- **Real‑time APIs** via Alpha Vantage:
  - Market status
  - Top gainers/losers
  - News & sentiment
  - Company fundamentals
- **Local SQLite database** (`stocks.db`) containing S&P 500‑style metadata:
  - Ticker, company, sector, industry, market‑cap bucket, etc.

**Single‑Agent Mode**

- One agent with access to all tools:
  - `get_tickers_by_sector`, `get_price_performance`, `get_company_overview`,
    `get_market_status`, `get_top_gainers_losers`, `get_news_sentiment`, `query_local_db`.
- System prompt enforces:
  - Tool usage for **current** or **numeric** data.
  - No fabrication when tools return errors.
- Suitable for straightforward finance questions like:
  - “What is NVIDIA’s 1‑year return?”
  - “Give me a quick overview of AAPL fundamentals.”

**Multi‑Agent Mode**

Pipeline of specialist agents:

- **Market Specialist**
  - Handles returns, performance, sectors/industries, and ranking by returns.
- **Fundamentals Specialist**
  - Focuses on P/E, EPS, market cap, 52‑week high/low.
- **Sentiment Specialist**
  - Uses news sentiment tools to summarize recent headlines per ticker.
- Additional logic:
  - Extracts and resolves tickers from conversational context.
  - Chooses which specialists to run based on the question.
  - Merges their outputs into a concise, aligned final answer.

**Streamlit UI**

- Chat‑style interface supporting:
  - Persistent conversation history.
  - Choice between **Single Agent** and **Multi‑Agent** in the sidebar.
  - Choice of OpenAI models (`gpt-4o-mini` vs `gpt-4o`).
- Shows which tools and agents were used for each answer, helping explain the system’s reasoning.

**Key skills**

- Tool‑calling and structured function schemas with the OpenAI Chat Completions API.
- Designing and debugging multi‑agent pipelines for a domain‑specific task (FinTech).
- Building a usable, stateful Streamlit chat UI with conversation memory.

---

### Repository Structure (High Level)

- `MP1_P1/` — Mini Project 1, Part 1 (embeddings & search demo).
- `MP1_P2/` — Mini Project 1, Part 2 (extended experiments and datasets).
- `MP2_P1P2/` — Mini Project 2, Parts 1–2 (foundations for retrieval & agents).
- `MP2_P3P4/` — Mini Project 2, Parts 3–4 (multi‑agent chatbot and evaluation).
- `MP3/` — Mini Project 3 (FinTech single‑ and multi‑agent chat app).

Each subfolder contains its own notebooks and/or app scripts, plus local `requirements.txt` where needed.

---

### How to Run (High Level)

1. **Create a virtual environment** and install dependencies per project, for example:
   ```bash
   cd MP3
   pip install -r requirements.txt
   ```
2. **Set environment variables**:
   - `OPENAI_API_KEY` (required for all LLM‑based components).
   - `PINECONE_API_KEY` and `INDEX_NAME` (for MP2 retrieval).
   - `ALPHAVANTAGE_API_KEY` (for MP3 FinTech tools).
3. **Launch Streamlit apps** (for UI‑based projects):
   - Example (MP1 / MP3):
     ```bash
     streamlit run app.py
     ```
4. **Run notebooks** (`.ipynb`) in your preferred environment (VS Code, Jupyter, or Colab) for experiments and evaluation scripts.

---

### Summary

Across these three projects, I progressed from **basic embeddings and similarity** (Project 1), to a **safety‑aware multi‑agent chatbot and evaluation framework** (Project 2), and finally to a **domain‑specific, tool‑using financial assistant with both single‑agent and multi‑agent architectures** (Project 3). Together, they showcase practical skills in LLM orchestration, retrieval, evaluation, and user‑facing app development.