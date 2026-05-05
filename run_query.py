"""
Query the knowledge base.

Edit the paths and query below, then run:
    python run_query.py
"""

from logic import query_knowledge_base

# HYPERPARAMETER

QUERY = "What are the factors that make people at risk for HNC?"
TOP_K = 5 # How many contexts should be returned

FAISS_PATH = "EXISTING_KNOWLEDGE_BASE.idx"
METADATA_PATH = "EXISTING_KNOWLEDGE_BASE.pkl"

MODEL_NAME = "all-MiniLM-L6-v2"

if __name__ == "__main__":
    results = query_knowledge_base(
        query=QUERY,
        faiss_path=FAISS_PATH,
        metadata_path=METADATA_PATH,
        model_name=MODEL_NAME,
        top_k=TOP_K,
    )

    print(f"\nQuery: {QUERY}\n")
    print("=" * 70)
    for i, r in enumerate(results, 1):
        print(f"\n[{i}] {r['source']} — page {r['page']}")
        print("-" * 70)
        print(r["paragraph"])