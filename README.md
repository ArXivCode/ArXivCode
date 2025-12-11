# ArXivCode: From Theory to Implementation

Bridge the gap between AI research and practical implementation. Search for theoretical concepts from arXiv papers and retrieve relevant code implementations with explanations.

## System Overview

ArXivCode uses **CodeBERT embeddings** with a **hybrid retrieval system** to find relevant code snippets from ML/AI research paper implementations.

### Key Features

- **Custom CodeBERT Embeddings**: 768-dimensional dense vectors generated using Microsoft's CodeBERT model
- **Hybrid Scoring**: Combines semantic similarity (60%) with keyword matching (40%) for improved relevance
- **Keyword Expansion**: Boosts function name matches (5x), paper title matches (4x), and code content matches (3x)
- **2,490 Curated Code Snippets**: Cleaned and filtered from 50+ influential ML papers
- **Interactive Web Interface**: Streamlit frontend with search, code display, and explanations

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────┐
│              Hybrid Retrieval System                │
├─────────────────────────────────────────────────────┤
│                                                     │
│  1. Semantic Search (60% weight)                    │
│     - Encode query with CodeBERT                    │
│     - Cosine similarity against 2,490 embeddings    │
│     - FAISS index for fast nearest neighbor search  │
│                                                     │
│  2. Keyword Matching (40% weight)                   │
│     - Function name matching (5x boost)             │
│     - Paper title matching (4x boost)               │
│     - Code content matching (3x boost)              │
│                                                     │
└─────────────────────────────────────────────────────┘
    │
    ▼
Ranked Results → Streamlit UI → User
```

## Quick Setup

### 1. Environment Setup

```bash
# Create virtual environment (Python 3.11)
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
cp .env.example .env
# Edit .env and add your keys:
# - GITHUB_TOKEN (for data collection)
# - OPENAI_API_KEY (for explanations)
```

### 3. Start the System

```bash
# Activate virtual environment
source venv/bin/activate

# Terminal 1: Start the API backend
python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000

# Terminal 2: Start the Streamlit frontend (in a new terminal)
source venv/bin/activate
streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0
```

Visit http://localhost:8501 to use the web interface.

## Embedding Generation

We generated custom embeddings using Microsoft's CodeBERT model with an enhanced strategy:

```python
# Each code snippet is embedded as:
embedding_text = f"{paper_title} {function_name} {code_text}"
```

This combines:
- **Paper context**: Links code to its research paper
- **Function name**: Captures the semantic intent of the function
- **Code content**: The actual implementation

### Embedding Statistics

| Metric | Value |
|--------|-------|
| Model | microsoft/codebert-base |
| Embedding Dimension | 768 |
| Total Snippets | 2,490 |
| Storage | ~7.3 MB (embeddings) |
| Index Type | FAISS FlatIP (cosine similarity) |

## Data Pipeline

```
Papers List → Download Repos → Extract Functions → Clean & Filter → Generate Embeddings
```

1. **Curated Papers**: Hand-selected influential ML papers (LoRA, BERT, GPT, etc.)
2. **Code Download**: Clone associated GitHub repositories
3. **Function Extraction**: Parse Python files to extract functions/classes
4. **Data Cleaning**: Filter irrelevant code (tests, configs, utilities)
5. **Embedding Generation**: CodeBERT encodes each snippet

### Data Cleaning Results

| Stage | Count | Reduction |
|-------|-------|-----------|
| Raw snippets | 37,000+ | - |
| After cleaning | 2,490 | 93.3% |

Filtering criteria:
- Remove test files and configuration code
- Keep only paper-relevant implementations
- Require meaningful function names
- Filter out utility/helper code

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/search` | POST | Search code snippets |
| `/explain` | POST | Generate code explanation |
| `/stats` | GET | System statistics |

### Search Example

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "how to implement LoRA", "top_k": 5}'
```

## Project Structure

```
arxivcode/
├── src/
│   ├── api/                 # FastAPI backend
│   │   └── app.py
│   ├── data_collection/     # Paper & code collection
│   │   ├── curated_papers_list.py
│   │   ├── code_downloader.py
│   │   ├── extract_snippets.py
│   │   └── clean_dataset.py
│   ├── embeddings/          # CodeBERT embedding generation
│   │   ├── code_encoder_model.py
│   │   └── generate_improved_embeddings.py
│   ├── retrieval/           # Hybrid retrieval system
│   │   ├── dense_retrieval.py
│   │   ├── faiss_index.py
│   │   └── cross_encoder_reranker.py
│   └── models/              # LLM explanation
│       └── explanation_llm.py
├── frontend/
│   └── app.py               # Streamlit UI
├── data/
│   └── processed/
│       ├── code_snippets_cleaned.json
│       └── embeddings_v2/
│           ├── code_embeddings.npy
│           └── metadata.json
└── tests/
```

## Technology Stack

| Component | Technology |
|-----------|------------|
| Embedding Model | Microsoft CodeBERT (768-dim) |
| Vector Search | FAISS (Facebook AI Similarity Search) |
| Optional Reranker | MS MARCO MiniLM Cross-Encoder |
| Backend API | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Language | Python 3.11 |

## Example Queries

- "how to implement LoRA"
- "transformer attention mechanism"
- "BERT fine-tuning"
- "flash attention"
- "PPO reinforcement learning"
- "vision transformer"
- "knowledge distillation"

## Requirements

- Python 3.11 (or 3.9+)
- 8GB+ RAM (16GB recommended)
- GitHub Personal Access Token (for data collection)
- Hugging Face account (for CodeBERT model access)

## Documentation

- [Data Collection Guide](docs/DATA_COLLECTION_GUIDE.md)
- [Model Setup Guide](docs/PAPER_COMPREHENSION_MODEL.md)
- [Implementation Summary](docs/IMPLEMENTATION_SUMMARY.md)

## License

MIT
