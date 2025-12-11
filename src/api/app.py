from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.retrieval.dense_retrieval import DenseRetrieval

app = FastAPI(title="ArXivCode API")

# Global retriever instance
retriever = None

@app.on_event("startup")
async def load_resources():
    global retriever
    print("Loading retrieval system...")
    retriever = DenseRetrieval(
        embedding_model_name="microsoft/codebert-base",
        use_gpu=False
    )
    stats = retriever.get_statistics()
    print(f"Loaded {stats['total_vectors']} code snippets")

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    use_reranker: bool = False
    hybrid_scoring: bool = True

class ExplainRequest(BaseModel):
    query: str
    code_snippet: str
    paper_title: str
    paper_context: str = ""

@app.get("/")
async def health_check():
    stats = retriever.get_statistics() if retriever else {}
    return {
        "status": "healthy",
        "total_snippets": stats.get('total_vectors', 0)
    }

@app.post("/search")
async def search(request: SearchRequest):
    if not retriever:
        raise HTTPException(status_code=503, detail="Retriever not initialized")
    
    # Use the DenseRetrieval system
    results = retriever.retrieve(
        query=request.query,
        top_k=request.top_k,
        use_reranker=request.use_reranker,
        hybrid_scoring=request.hybrid_scoring
    )
    
    # Format results for frontend
    formatted_results = []
    for r in results:
        meta = r['metadata']
        formatted_results.append({
            "paper_title": meta.get('paper_title', 'Unknown'),
            "code_text": meta.get('code_text', ''),
            "function_name": meta.get('function_name', 'Unknown'),
            "file_path": meta.get('file_path', ''),
            "paper_url": meta.get('paper_url', ''),
            "repo_url": meta.get('repo_url', ''),
            "score": r['score'],
            "rank": r.get('rank', 0)
        })
    
    return {"query": request.query, "results": formatted_results}

@app.post("/explain")
async def explain(request: ExplainRequest):
    # Placeholder for LLM integration
    explanation = f"""**Code Explanation for: {request.paper_title}**

This code snippet implements functionality related to "{request.query}".

The function/class shown is part of the implementation from the paper "{request.paper_title}".

*Note: Full LLM-powered explanations coming soon!*"""
    
    return {"explanation": explanation}

@app.get("/stats")
async def stats():
    if not retriever:
        return {"total_snippets": 0, "embedding_dim": 768, "model": "not loaded"}
    
    stats = retriever.get_statistics()
    return {
        "total_snippets": stats['total_vectors'],
        "embedding_dim": stats['embedding_dim'],
        "model": stats['embedding_model']
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
