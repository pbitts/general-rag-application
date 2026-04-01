# RAG System — Intelligent Document Q&A with Automatic Change Detection

A production-ready Retrieval-Augmented Generation (RAG) pipeline that enables natural language querying over PDF documents. The system automatically detects document and configuration changes to decide whether re-indexing is needed, supports multi-turn conversational memory, and integrates with LangSmith for observability.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Key Features](#key-features)
- [Techniques & Stack](#techniques--stack)
- [How It Works](#how-it-works)
- [Configuration](#configuration)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Project Structure](#project-structure)

---

## Overview

This system answers questions grounded exclusively in the content of your PDF documents. Rather than relying on a language model's general knowledge, every response is retrieved from and constrained to the indexed document corpus — reducing hallucinations and ensuring traceability back to source pages.

It is designed for both **technical teams** (who need reliable, auditable AI outputs) and **analyst teams** (who want to query large document repositories without manual searching).

---

## Architecture

```
PDF Files
    │
    ▼
DocumentLoader          ← Loads single or multiple PDFs via PyPDFLoader
    │
    ▼
RecursiveCharacterTextSplitter  ← Splits pages into overlapping chunks
    │
    ▼
HuggingFaceEmbeddings   ← Converts chunks to vector representations
    │
    ▼
ChromaDB (Vector Store) ← Persists and retrieves embeddings locally
    │
    ▼
Retriever               ← Fetches top-K most relevant chunks per query
    │
    ▼
ChatPromptTemplate      ← Injects context + conversation history
    │
    ▼
ChatGroq (LLM)          ← Generates grounded answers via Groq API
    │
    ▼
ConversationMemory      ← Stores recent exchanges for multi-turn continuity
```

---

## Key Features

**Smart Re-indexing Detection**
The `MetadataManager` computes an MD5 hash of the document corpus and compares it — along with chunking parameters and the embedding model name — against a stored snapshot. Re-indexing only occurs when something has actually changed, keeping startup fast for repeated runs.

**Multi-turn Conversation Memory**
A sliding-window `ConversationMemory` retains the last N question–answer pairs (configurable via `CONVERSATION_MEMORY_SIZE`). This context is injected into every prompt, enabling coherent follow-up questions.

**Flexible Retrieval Strategies**
Three retrieval modes are supported and selectable via configuration: standard similarity search, Maximum Marginal Relevance (MMR) for diversity-aware retrieval, and similarity with a minimum score threshold.

**Source Attribution**
When `SHOW_SOURCES` is enabled, each answer is accompanied by the source document name and page number, along with a content preview — giving analysts a direct path back to the original material.

**LangSmith Observability**
The system integrates with LangSmith for full tracing of retrieval and generation steps, enabling debugging and performance monitoring of the RAG chain.

---

## Techniques & Stack

| Layer | Technology | Purpose |
|---|---|---|
| Document Loading | `langchain-community` PyPDFLoader | Parse PDF pages with metadata |
| Text Splitting | `RecursiveCharacterTextSplitter` | Chunk text with configurable size and overlap |
| Embeddings | `HuggingFaceEmbeddings` (CPU) | Dense vector representations of chunks |
| Vector Store | `ChromaDB` (persistent) | Local similarity search over embeddings |
| LLM | `ChatGroq` | Fast inference for answer generation |
| Prompt Design | `ChatPromptTemplate` + `MessagesPlaceholder` | Structured, context-aware prompting |
| Memory | Sliding-window `deque` | Bounded conversation history |
| Change Detection | MD5 hashing + JSON metadata | Avoid unnecessary re-indexing |
| Observability | LangSmith | Chain tracing and debugging |

---

## How It Works

### 1. Indexing Phase

On startup, the system checks whether re-indexing is required by comparing the current MD5 hash of the document directory (and key config values) against a stored metadata file. If changes are detected — or if no index exists — it proceeds to:

1. Load all PDFs from the configured path using `PyPDFLoader`
2. Split each page into overlapping text chunks using `RecursiveCharacterTextSplitter`
3. Embed all chunks using a HuggingFace sentence-transformer model
4. Persist the resulting vector index to ChromaDB

If nothing has changed, the existing index is loaded directly, skipping steps 1–4.

### 2. Query Phase

For each question in the configured question list:

1. The retriever searches ChromaDB for the top-K most semantically similar chunks
2. Retrieved chunks are formatted and injected into the system prompt as `{context}`
3. The recent conversation history is injected via `{chat_history}`
4. The LLM generates a response grounded in the context
5. The question–answer pair is saved to `ConversationMemory` for use in subsequent turns
6. Source metadata (document name, page number, content preview) is displayed if configured

---

## Configuration

All settings are managed through the `Settings` class (from `config.py`). Key parameters:

| Parameter | Description |
|---|---|
| `DOCUMENTS_PATH` | Path to a single PDF or directory of PDFs |
| `CHROMA_DB_PATH` | Directory where the vector index is persisted |
| `METADATA_PATH` | Path to the metadata snapshot JSON file |
| `EMBEDDING_MODEL` | HuggingFace model name for embeddings |
| `LLM_MODEL` | Groq model identifier |
| `GROQ_API_KEY` | API key for Groq inference |
| `CHUNK_SIZE` | Character length of each text chunk |
| `CHUNK_OVERLAP` | Character overlap between adjacent chunks |
| `SEARCH_TYPE` | `similarity`, `mmr`, or `similarity_score_threshold` |
| `SEARCH_K` | Number of chunks to retrieve per query |
| `FETCH_K` | Candidate pool size for MMR (MMR only) |
| `LAMBDA_MULT` | Diversity weight for MMR (0 = max diversity, 1 = max relevance) |
| `SCORE_THRESHOLD` | Minimum similarity score for threshold search |
| `CONVERSATION_MEMORY_SIZE` | Number of past exchanges to retain |
| `QUESTIONS` | List of questions to process automatically |
| `FORCE_REINDEX` | Force re-indexing regardless of change detection |
| `VERBOSE` | Enable detailed logging |
| `SHOW_SOURCES` | Display source documents and pages with answers |
| `SHOW_RETRIEVAL_DOCS` | Log retrieval step details |

---

## Getting Started

### Prerequisites

- Python 3.9+
- A [Groq API key](https://console.groq.com/)
- (Optional) A [LangSmith](https://smith.langchain.com/) account for observability

### Installation

```bash
git clone <repository-url>
cd <repository-directory>
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here

# Optional — LangSmith observability
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT=your_project_name
```

### Place Your Documents

Put your PDF files in the directory specified by `DOCUMENTS_PATH` in your config (default: `./documents/`):

```
documents/
├── report_q1.pdf
├── technical_spec.pdf
└── policy_manual.pdf
```

---

## Usage

### Run with the Default Question List

Configure your questions in `config.py` (the `QUESTIONS` list), then run:

```bash
python main.py
```

The system will print a startup banner, index documents (or load the existing index), process each question, and display a summary.

### Run the Web Interface
```bash
uvicorn api:app --reload --port 8000
```

Access `http://localhost:8000` in your browser. Configure the document path,
splitter and retriever settings, then chat directly with your documents.

### Example Output

```
============================================================
RAG SYSTEM — AUTOMATIC CHANGES DETECTION
============================================================

Changes detected on docs or configs
============================================================
Docs Indexing
============================================================
Loading: report_q1.pdf
 "42" pages loaded
...
 "186" chunks

============================================================
PROCESSING QUESTIONS
============================================================

Question 1/2
============================================================
❓ What were the main risks identified in Q1?
------------------------------------------------------------
Answer:
The Q1 report identifies three primary risk areas: supply chain disruption...

------------------------------------------------------------
Sources:

[1] Page 7 - report_q1.pdf
    The supply chain risk assessment conducted in January highlighted...

============================================================
 Execution Summary
============================================================
📄 Docs: ./documents/
🤖 Model: llama3-70b-8192
🔍 Search: mmr (K=4)
✂️  Chunks: 1000 chars (overlap=200)
❓ Processed Questions: 2
🧠 Conversations in Memory: 2
============================================================
```

### Force Re-indexing

To re-index regardless of the change detection result, set `FORCE_REINDEX=True` in your config or environment.

---

## Project Structure

```
.
├── main.py              # Entry point; orchestrates indexing and Q&A
├── rag_system.py        # Core classes: RAGSystem, DocumentLoader, MetadataManager, ConversationMemory
├── api.py               # FastAPI backend (web interface)
├── config.py            # Settings dataclass with all configurable parameters
├── observability.py     # LangSmith configuration
├── documents/           # Default directory for PDF input files
├── chroma_db/           # Persisted ChromaDB vector index (auto-created)
├── metadata/
│   └── metadata.json    # Change-detection snapshot (auto-created)
├── ui/
│   └── index.html       # Web interface frontend
└── requirements.txt
```

---

## Notes for Analysts

- All answers are grounded in the indexed documents. The LLM is instructed not to draw on outside knowledge.
- Enable `SHOW_SOURCES=True` to see exactly which page and document each answer comes from.
- You can ask follow-up questions naturally — the system remembers the last several exchanges.
- If you add or update PDF files, the system will detect the change on the next run and re-index automatically.

## Notes for Engineers

- The embedding model runs on CPU by default (`device: cpu`). For GPU acceleration, change `model_kwargs={'device': 'cuda'}` in `_get_embedding_model`.
- ChromaDB is persisted locally. For multi-user or cloud deployments, consider replacing it with a hosted vector database.
- The RAG chain is built with LangChain's `RunnablePassthrough` pattern, making it straightforward to extend with additional steps (e.g., query rewriting, re-ranking).
- LangSmith tracing is initialized in `observability.py` before `main()` runs, capturing the full chain execution.
- The core logic lives in rag_system.py and is shared between the CLI (main.py) and the web backend (api.py) — both import RAGSystem independently without interfering with each other.