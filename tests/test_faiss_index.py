#!/usr/bin/env python3
"""
Test FAISS index built from CodeBERT embeddings.
This script loads the FAISS index and tests retrieval with sample queries.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from retrieval.faiss_index import FAISSIndexManager
from embeddings.code_encoder_model import CodeEncoder
from embeddings.generate_code_embeddings import generate_embedding
import numpy as np
import torch


def test_faiss_index(
    index_path: str = "data/processed/faiss_index.index",
    metadata_path: str = "data/processed/faiss_metadata.pkl",
    embedding_dim: int = 768
):
    """
    Test the FAISS index with sample queries.
    
    Args:
        index_path: Path to FAISS index file
        metadata_path: Path to metadata pickle file
        embedding_dim: Embedding dimension (768 for CodeBERT-base)
    """
    print("\n" + "="*70)
    print("TESTING FAISS INDEX")
    print("="*70)
    
    # 1. Load FAISS index
    print(f"\n1. Loading FAISS index...")
    print(f"   Index: {index_path}")
    print(f"   Metadata: {metadata_path}")
    
    index_manager = FAISSIndexManager(
        embedding_dim=embedding_dim,
        index_type="FlatIP"
    )
    
    try:
        index_manager.load(index_path, metadata_path)
        stats = index_manager.get_stats()
        print(f"   ✅ Index loaded successfully!")
        print(f"   Total vectors: {stats['total_vectors']}")
        print(f"   Embedding dimension: {stats['embedding_dim']}")
        print(f"   Index type: {stats['index_type']}")
    except Exception as e:
        print(f"   ❌ Error loading index: {e}")
        return
    
    # 2. Load CodeBERT encoder for query encoding
    print(f"\n2. Loading CodeBERT encoder for queries...")
    try:
        encoder = CodeEncoder(
            model_name="microsoft/codebert-base",
            max_length=512,
            device=None  # Auto-detect
        )
        encoder.model.eval()
        print(f"   ✅ CodeBERT encoder loaded")
    except Exception as e:
        print(f"   ❌ Error loading encoder: {e}")
        return
    
    # 3. Test queries
    test_queries = [
        "implement transformer attention mechanism",
        "fine-tune large language model",
        "parameter efficient training methods",
        "masked language modeling",
        "contrastive learning",
        "how to implement LoRA",
        "self-attention layer code",
        "tokenizer implementation",
        "batch normalization",
        "graph neural network",
    ]
    
    print(f"\n3. Testing retrieval with {len(test_queries)} queries...")
    print("-" * 70)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n📝 Query {i}: '{query}'")
        print("-" * 70)
        
        try:
            # Encode query using CodeBERT
            query_embedding = generate_embedding(encoder, query, max_length=512)
            
            # Search index
            results = index_manager.search(query_embedding, top_k=5)
            
            if not results:
                print("   ⚠️  No results found")
                continue
            
            # Display results
            for result in results:
                meta = result['metadata']
                print(f"\n   Rank {result['rank']} (Score: {result['score']:.4f})")
                
                # Display available metadata fields
                if 'paper_title' in meta:
                    print(f"   📄 Paper: {meta.get('paper_title', 'N/A')[:60]}...")
                if 'paper_id' in meta:
                    print(f"   🆔 Paper ID: {meta.get('paper_id', 'N/A')}")
                if 'repo_name' in meta:
                    print(f"   🔗 Repo: {meta.get('repo_name', 'N/A')}")
                if 'file_path' in meta:
                    print(f"   📁 File: {meta.get('file_path', 'N/A')}")
                if 'function_name' in meta:
                    print(f"   ⚙️  Function: {meta.get('function_name', 'N/A')}")
                if 'code_file_path' in meta:
                    print(f"   📄 Code File: {meta.get('code_file_path', 'N/A')}")
        
        except Exception as e:
            print(f"   ❌ Error processing query: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 4. Statistics
    print(f"\n4. Index Statistics")
    print("-" * 70)
    stats = index_manager.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # 5. Test with a code snippet query
    print(f"\n5. Testing with code snippet query...")
    print("-" * 70)
    
    code_query = """
def attention(q, k, v, mask=None):
    scores = torch.matmul(q, k.transpose(-2, -1))
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    attention_weights = F.softmax(scores, dim=-1)
    return torch.matmul(attention_weights, v)
"""
    
    print(f"📝 Code Query: 'attention mechanism implementation'")
    try:
        query_embedding = generate_embedding(encoder, code_query, max_length=512)
        
        results = index_manager.search(query_embedding, top_k=3)
        
        if results:
            print(f"   ✅ Found {len(results)} results")
            for result in results[:2]:
                meta = result['metadata']
                print(f"\n   Rank {result['rank']} (Score: {result['score']:.4f})")
                if 'paper_title' in meta:
                    print(f"   📄 {meta.get('paper_title', 'N/A')[:50]}...")
                if 'repo_name' in meta:
                    print(f"   🔗 {meta.get('repo_name', 'N/A')}")
        else:
            print("   ⚠️  No results found")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n" + "="*70)
    print("✅ TESTING COMPLETE")
    print("="*70)


def main():
    """Main function with command-line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test FAISS index built from CodeBERT embeddings"
    )
    parser.add_argument(
        "--index_path",
        type=str,
        default="data/processed/faiss_index.index",
        help="Path to FAISS index file"
    )
    parser.add_argument(
        "--metadata_path",
        type=str,
        default="data/processed/faiss_metadata.pkl",
        help="Path to metadata pickle file"
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=768,
        help="Embedding dimension (default: 768 for CodeBERT-base)"
    )
    
    args = parser.parse_args()
    
    test_faiss_index(
        index_path=args.index_path,
        metadata_path=args.metadata_path,
        embedding_dim=args.embedding_dim
    )


if __name__ == "__main__":
    main()
