"""
Build a knowledge base from a list of PDFs.

Edit the paths below, then run:
    python run_build.py
"""

from logic import extract_paragraphs_from_pdfs, build_knowledge_base

# HYPERPARAMETERS
## Change based on where your source files are actually located.

PDF_PATHS = [
    "FILE1.pdf",
    "FILE2.pdf",
    "FILE3.pdf",
]

OUTPUT_DIR = "./"
CSV_PATH = f"{OUTPUT_DIR}/dental_knowledge_base.csv"
FAISS_PATH = f"{OUTPUT_DIR}/dental_knowledge_base.idx"
METADATA_PATH = f"{OUTPUT_DIR}/dental_knowledge_base.pkl"

MODEL_NAME = "all-MiniLM-L6-v2"

if __name__ == "__main__":
    records = extract_paragraphs_from_pdfs(PDF_PATHS)

    build_knowledge_base(
        records=records,
        csv_path=CSV_PATH,
        faiss_path=FAISS_PATH,
        metadata_path=METADATA_PATH,
        model_name=MODEL_NAME,
    )