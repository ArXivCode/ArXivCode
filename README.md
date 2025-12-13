# ArXivCode: From Theory to Implementation

Bridge the gap between AI research and practical implementation. Search for theoretical concepts from arXiv papers and retrieve relevant code implementations with explanations.

**Deployment Link** — https://arxivcode-frontend-215017069058.us-central1.run.app/ *(**Note:** the search UI may take 1–2 minutes to appear due to Cloud Run cold start.)*

## System Overview

ArXivCode uses **custom CodeBERT embeddings** with a **hybrid retrieval system** to find relevant code snippets from ML/AI research paper implementations.

## Key Features

- **Custom CodeBERT Embeddings**: 768-dimensional dense vectors generated using Microsoft's CodeBERT model
- **Hybrid Scoring**: Combines semantic similarity (60%) with keyword matching (40%) for improved relevance
- **Keyword Expansion**: Boosts function name matches (5x), paper title matches (4x), and code content matches (3x)
- **2,490 Curated Code Snippets**: Cleaned and filtered from 196 influential ML papers
- **Interactive Web Interface**: Streamlit frontend with search, code display, and explanations


## Example Queries

- "how to implement LoRA"
- "transformer attention mechanism"
- "BERT fine-tuning"
- "flash attention"
- "PPO reinforcement learning"
- "vision transformer"
- "knowledge distillation"

## Documentation

### Getting Started
- **[Tutorial](Tutorial.md)** - Complete setup and usage guide
- **[Deployment Link](https://arxivcode-frontend-215017069058.us-central1.run.app/)** - ***Note:** the search UI may take 1–2 minutes to appear due to Cloud Run cold start.*

### Project Documentation
- **[Project Proposal](Proposal.md)** - Original project proposal and objectives
- **[Final Report](Final%20Report.md)** - Complete project report with methodology, results, and evaluation

### Technical Documentation
- [Data Collection Guide](docs/DATA_COLLECTION_GUIDE.md)
- [Model Setup Guide](docs/PAPER_COMPREHENSION_MODEL.md)
- [Implementation Summary](docs/IMPLEMENTATION_SUMMARY.md)

### Evaluation & Results
- **[Results Generation Script](results/README_RESULTS.md)** - Generate evaluation metrics and results for the Final Report




