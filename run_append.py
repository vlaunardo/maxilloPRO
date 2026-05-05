"""
Append entries from a manually-curated CSV into an existing knowledge base.

The CSV must have columns: source, page, paragraph (case-insensitive).

Edit the paths below, then run:
    python run_append.py
"""

from logic import load_records_from_csv, append_to_knowledge_base

# HYPERPARAMETER

CSV_TO_APPEND = "PATH_TO_ANOTHER_KNOWLEDGE_BASE.csv"
FAISS_PATH = "EXISTING_KNOWLEDGE_BASE.idx"
METADATA_PATH = "EXISTING_KNOWLEDGE_BASE.pkl"

MODEL_NAME = "all-MiniLM-L6-v2"

if __name__ == "__main__":
    new_records = load_records_from_csv(CSV_TO_APPEND)

    append_to_knowledge_base(
        new_records=new_records,
        faiss_path=FAISS_PATH,
        metadata_path=METADATA_PATH,
        model_name=MODEL_NAME,
    )