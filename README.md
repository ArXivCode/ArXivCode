# ArxivCode

Machine learning project for paper-code understanding and retrieval.

## Project Structure

```
arxivcode/
├── src/                        # Source code
│   ├── data_collection/        # Data collection pipeline
│   │   ├── arxiv_github_collector.py  # Main collector
│   │   └── __init__.py
│   ├── models/                 # Model training & inference
│   ├── retrieval/              # Retrieval system
│   └── api/                    # Backend API
├── data/                       # Data storage
│   ├── raw/papers/             # Paper-code pairs
│   ├── processed/              # Cleaned data
│   └── metadata/               # Logs, statistics
├── tests/                      # Tests
│   └── test_arxiv_github.py
├── docs/                       # Documentation
│   └── setup/
│       └── DATA_COLLECTION_GUIDE.md
├── .env.example
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Setup Environment

```bash
# Create and activate virtual environment (Python 3.11)
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure GitHub Token

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your GitHub token
# Get token at: https://github.com/settings/tokens
# Required scopes: public_repo, read:org
```

### 3. Collect Paper-Code Pairs

We provide three automated collection methods:

**Option A: Automated GitHub Discovery** (Recommended - Fully Automated)
```bash
# Automatically discovers papers from GitHub trending ML/AI repos
python src/data_collection/pwc_dataset_collector.py
```
- Searches GitHub for popular ML/AI repos with ArXiv references
- Extracts paper-code pairs automatically from README files
- No manual curation required - fully sustainable

**Option B: Curated High-Impact Papers** (Fast & Reliable)
```bash
# Uses a curated list of 215+ high-impact papers
python src/data_collection/pwc_hf_collector.py
```
- Includes landmark papers: BERT, GPT-3, LLaMA, CLIP, etc.
- Fast - no API rate limits
- **Note**: While the initial list was manually curated, you can expand it by adding new papers to [curated_papers_list.py](src/data_collection/curated_papers_list.py) as they become popular
- Validates all repos against GitHub API for current stats

**Option C: ArXiv + GitHub Search** (Comprehensive but slower)
```bash
# Searches ArXiv and matches with GitHub repos
python src/data_collection/arxiv_github_collector.py
```
Note: May hit ArXiv API rate limits. Wait 15+ minutes between runs.

**Output**: `data/raw/papers/paper_code_pairs.json`

**Collected So Far**: 153 paper-code pairs
- Average Stars: 17,790
- Year Range: 2013-2023
- Papers: BERT, GPT-3, LLaMA 2, CLIP, Mistral, Mamba, LoRA, QLoRA, FlashAttention, DPO, Whisper, etc.

## Retrieval System Testing

The retrieval system uses FAISS for efficient similarity search across paper-code pairs. Test the system by running these commands in order:

```bash
# 1. Test imports
python -c "from src.retrieval import FAISSIndexManager, DenseRetrieval; print('✅ Imports work!')"

# 2. Test FAISS manager
python -m src.retrieval.faiss_index

# 3. Build index with real data
python -m src.retrieval.build_index --input data/raw/papers/paper_code_pairs.json

# 4. Test retrieval
python -m src.retrieval.test_retrieval
```

**Index Storage**: `data/processed/FAISS/`
- `faiss_index.index` - Vector similarity index
- `faiss_index_vectorizer.pkl` - TF-IDF vectorizer (for CPU-stable embeddings)
- `faiss_metadata.pkl` - Metadata for retrieved results

**Features**:
- TF-IDF embeddings for stable, CPU-friendly similarity search
- Repository-level code retrieval
- Query filtering by stars, year, and topics
- Validated on 153 paper-code pairs with strong relevance

## Current Status

✅ **Phase 1: Data Collection** (Complete)
- Papers With Code integration
- ArXiv API integration
- GitHub repository search
- Filtering & metadata collection

✅ **Phase 2: Retrieval System** (Complete)
- FAISS indexing with TF-IDF embeddings
- Dense retrieval pipeline
- Query testing and validation

⏳ **Phase 3: Data Processing** (Upcoming)
- Function-level code snippet extraction
- Cross-encoder re-ranking

⏳ **Phase 4: API & Frontend** (Upcoming)
- Backend API
- Web interface

## Documentation

- **Data Collection Guide**: [docs/setup/DATA_COLLECTION_GUIDE.md](docs/setup/DATA_COLLECTION_GUIDE.md)

## Requirements

- Python 3.11
- GitHub Personal Access Token (recommended for data collection)

## License

MIT
