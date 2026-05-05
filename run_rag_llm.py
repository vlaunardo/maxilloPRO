"""
Run the RAG pipeline: retrieve from the knowledge base + ask the LLM.

Before running, set your OpenAI API key:
    export OPENAI_API_KEY="sk-..."

Then edit the query and paths below and run:
    python run_rag.py
"""

from llm import make_client, run_rag

# HYPERPARAMETER

QUERY = "How can I care for Mandibular Malposition After Bony Reconstruction?"

FAISS_PATH = "EXISTING_KNOWLEDGE_BASE.idx"
METADATA_PATH = "EXISTING_KNOWLEDGE_BASE.pkl"

# k=0 sends the query to the LLM with no retrieved context (baseline).
TOP_K = 5
INCLUDE_CONTEXT = True

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "gpt-4o-2024-11-20"

SYSTEM_PROMPT = (
    "Please tailor your response to be comprehensible, relevant, and empathetic."
)

if __name__ == "__main__":
    client = make_client()  # reads OPENAI_API_KEY from env

    result = run_rag(
        query=QUERY,
        faiss_path=FAISS_PATH,
        metadata_path=METADATA_PATH,
        client=client,
        embedding_model=EMBEDDING_MODEL,
        llm_model=LLM_MODEL,
        system_prompt=SYSTEM_PROMPT,
        top_k=TOP_K,
        include_context=INCLUDE_CONTEXT,
    )

    print(result)