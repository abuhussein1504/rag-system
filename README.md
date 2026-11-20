# RAG System

A modular Retrieval-Augmented Generation (RAG) pipeline with an evaluation harness, built with Python.

## What is RAG?

RAG is an AI technique that improves LLM responses by first **retrieving** relevant documents from a knowledge base, then passing them as context to the model before **generating** an answer. This reduces hallucinations and keeps answers grounded in your actual data.

## Project Structure

```
rag-system/
├── main.py                            # FastAPI entry point & lifespan
├── pyproject.toml                     # Project metadata & configurations
├── uv.lock                            # Dependency lockfile
├── .env                               # Environment variables (ignored by git)
├── .env.example                       # Environment variables template
├── .gitignore
├── LICENSE
├── README.md
│
├── src/                               # Main source code
│   ├── config.py                      # Pydantic Settings & environment config
│   │
│   ├── models/                        # Core domain models
│   │   ├── enums.py                   # Enums (status, file types, sources)
│   │   └── schemas.py                 # Domain models (Chunk, Document, etc.)
│   │
│   ├── api/                           # API layer (FastAPI)
│   │   ├── routes/
│   │   │   ├── base.py                # Health / status endpoints
│   │   │   └── data.py                # Document upload & ingestion endpoints
│   │   └── schemas/
│   │       └── data.py                # API request/response schemas
│   │
│   ├── core/                          # Business logic
│   │   ├── ingestion/                 # Document ingestion pipeline
│   │   │   ├── loaders/               # File-type loaders
│   │   │   │   ├── pdf.py
│   │   │   │   ├── docx.py
│   │   │   │   ├── html.py
│   │   │   │   └── text.py
│   │   │   ├── cleaner.py             # Text cleaning & normalisation
│   │   │   ├── processing.py          # Orchestrates load → clean → chunk
│   │   │   ├── registry.py            # Loader registry / factory
│   │   │   └── storage.py             # File saving & validation logic
│   │   ├── chunking/                  # Chunking strategies
│   │   │   ├── base.py                # Abstract chunker interface
│   │   │   └── fixed.py               # Fixed-size chunker
│   │   ├── retrieval/                 # Retrieval logic (planned)
│   │   ├── generation/                # Generation logic (planned)
│   │   └── llm/                       # LLM client abstractions (planned)
│   │
│   └── infrastructure/                # Infrastructure & persistence adapters
│
├── scripts/                           # CLI commands & utility scripts
│
├── tests/                             # Test suite
│   ├── unit/
│   │   ├── test_schemas.py            # Unit tests for domain models
│   │   ├── test_upload.py             # Unit tests for upload endpoint
│   │   ├── test_loaders.py            # Unit tests for file loaders
│   │   └── test_processing.py         # Unit tests for ingestion pipeline
│   └── integration/                   # Integration tests
│
├── data/                              # Data directory (ignored by git)
│   ├── raw/                           # Raw uploaded files (by project_id)
│   ├── processed/                     # Parsed & chunked data
│   └── indexes/                       # Vector indices / local cache
│
├── configs/                           # Application & pipeline YAML configurations
│   ├── base.yaml
│   └── experiments/
│
├── eval/                              # Evaluation harness & benchmarks
│
└── docs/                              # Project documentation & architecture notes
    ├── architecture.md
    └── decisions/
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

### Run the API

```bash
uvicorn main:app --reload
```

## License

MIT

