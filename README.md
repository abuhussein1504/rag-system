# RAG System

A modular Retrieval-Augmented Generation (RAG) pipeline with an evaluation harness, built with Python.

## What is RAG?

RAG is an AI technique that improves LLM responses by first **retrieving** relevant documents from a knowledge base, then passing them as context to the model before **generating** an answer. This reduces hallucinations and keeps answers grounded in your actual data.

## Project Structure

```
rag-system/
├── src/rag/              # Core pipeline modules
│   ├── chunking/         # Split documents into chunks
│   ├── embedding/        # Convert chunks to vector embeddings
│   ├── indexing/         # Store & search vectors (Qdrant)
│   └── generation/       # Query LLM with retrieved context
├── api/                  # FastAPI REST API
├── scripts/              # CLI tools (ingest, eval)
├── configs/              # YAML configuration files
├── eval/                 # Evaluation golden set & reports
├── tests/
│   ├── unit/
│   └── integration/
└── data/
    ├── raw/              # Original documents
    ├── processed/        # Chunked & embedded documents
    └── indexes/          # Vector index files
```

## Tech Stack

| Component | Library |
|---|---|
| API | FastAPI + Uvicorn |

## Getting Started

### Prerequisites

- Python >= 3.12

### Installation

```bash
git clone https://github.com/abuhussein1504/rag-system.git
cd rag-system

# Create a virtual environment
uv venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies one by one as you need them
uv pip install fastapi uvicorn
uv pip install pydantic pydantic-settings
```

### Configuration

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

```env

```

### Run the API

```bash
uvicorn api:app --reload
```

## License

MIT

