"""
Core logic for building, extending, and querying a RAG knowledge base.

This file contains all the functions. Use the run_*.py scripts to actually
execute things.

Assumptions:
- PDFs are 2-column layout. Single-column PDFs will be parsed incorrectly.
- Embeddings use sentence-transformers (default: all-MiniLM-L6-v2).
- Vector store is FAISS IndexFlatL2 (exact search, fine up to ~100k paragraphs).
"""

import os
import csv
import pickle
from statistics import median

import numpy as np
import pdfplumber
import faiss
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------------------------

def extract_paragraphs_from_pdf(
    pdf_path,
    x_split_ratio=0.5,
    min_words=10,
    short_line_threshold=0.85,
):
    """
    Extract paragraphs from a 2-column PDF.

    Returns a list of (page_number, paragraph_text) tuples. Page numbers are
    1-indexed. Paragraphs shorter than `min_words` are dropped.
    """
    paragraphs = []

    def extract_paragraphs_from_column(lines, page_num):
        if not lines:
            return []

        line_widths = [line[-1]["x1"] - line[0]["x0"] for line in lines if line]
        mode_width = median(line_widths)

        para_list = []
        paragraph = ""

        for i, line in enumerate(lines):
            line_text = " ".join(w["text"] for w in line)
            line_width = line[-1]["x1"] - line[0]["x0"] if line else 0

            paragraph += " " + line_text

            is_last_line = (i == len(lines) - 1)
            is_short_line = (line_width < short_line_threshold * mode_width)

            if is_short_line and not is_last_line:
                if paragraph.strip() and len(paragraph.split()) >= min_words:
                    para_list.append((page_num + 1, paragraph.strip()))
                paragraph = ""

        if paragraph.strip() and len(paragraph.split()) >= min_words:
            para_list.append((page_num + 1, paragraph.strip()))

        return para_list

    def sort_words(words):
        return sorted(words, key=lambda w: (w["top"], w["x0"]))

    def group_lines(words):
        lines = []
        current_line = []
        last_top = None
        for word in sort_words(words):
            if last_top is None or abs(word["top"] - last_top) < 10:
                current_line.append(word)
            else:
                lines.append(current_line)
                current_line = [word]
            last_top = word["top"]
        if current_line:
            lines.append(current_line)
        return lines

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            split_x = page.width * x_split_ratio

            left_words, right_words = [], []
            for word in page.extract_words():
                if word["x0"] < split_x:
                    left_words.append(word)
                else:
                    right_words.append(word)

            lines_left = group_lines(left_words)
            lines_right = group_lines(right_words)

            paras_left = extract_paragraphs_from_column(lines_left, page_num)
            paras_right = extract_paragraphs_from_column(lines_right, page_num)

            paragraphs.extend(paras_left + paras_right)

    return paragraphs


def extract_paragraphs_from_pdfs(pdf_paths):
    """
    Run extraction over a list of PDFs. Returns a list of dicts with keys
    `source`, `page`, `paragraph` — the canonical record format used everywhere
    else in this module.
    """
    records = []
    for path in pdf_paths:
        print(f"Extracting from {path}")
        source = os.path.basename(path)
        for page, paragraph in extract_paragraphs_from_pdf(path):
            records.append({"source": source, "page": page, "paragraph": paragraph})
    return records


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------

# Canonical column names. Lowercase everywhere — keep these in sync with the
# dict keys in the records produced by extract_paragraphs_from_pdfs.
CSV_COLUMNS = ["source", "page", "paragraph"]


def save_records_to_csv(records, csv_path):
    """Write records (list of dicts) to CSV."""
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for r in records:
            writer.writerow({k: r[k] for k in CSV_COLUMNS})


def load_records_from_csv(csv_path):
    """
    Read records from CSV. Accepts both lowercase and uppercase headers so
    older CSVs still work.
    """
    records = []
    with open(csv_path, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # Build a case-insensitive header map.
        header_map = {h.lower(): h for h in reader.fieldnames or []}
        for required in CSV_COLUMNS:
            if required not in header_map:
                raise ValueError(
                    f"CSV at {csv_path} is missing required column '{required}'. "
                    f"Found columns: {reader.fieldnames}"
                )
        for row in reader:
            records.append({
                "source": row[header_map["source"]],
                "page": int(row[header_map["page"]]),
                "paragraph": row[header_map["paragraph"]],
            })
    return records


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def embed_texts(texts, model_name="all-MiniLM-L6-v2", model=None):
    """
    Embed a list of strings. Pass an already-loaded `model` to avoid reloading
    when calling this multiple times in one script.
    """
    if model is None:
        model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True)
    return np.array(embeddings)


# ---------------------------------------------------------------------------
# FAISS index + metadata (kept in sync as one knowledge base)
# ---------------------------------------------------------------------------

def build_index(embeddings):
    """Create a fresh FAISS IndexFlatL2 from an embeddings array."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index


def save_kb(index, metadata, faiss_path, metadata_path):
    """
    Save the FAISS index and metadata together. Always call this — never write
    one without the other, or the KB will get out of sync.
    """
    os.makedirs(os.path.dirname(faiss_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(metadata_path) or ".", exist_ok=True)
    faiss.write_index(index, faiss_path)
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)


def load_kb(faiss_path, metadata_path):
    """Load an existing knowledge base. Returns (index, metadata)."""
    index = faiss.read_index(faiss_path)
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata


# ---------------------------------------------------------------------------
# High-level operations: build, append, query
# ---------------------------------------------------------------------------

def build_knowledge_base(
    records,
    faiss_path,
    metadata_path,
    csv_path=None,
    model_name="all-MiniLM-L6-v2",
):
    """
    Build a knowledge base from scratch given a list of records (dicts with
    `source`, `page`, `paragraph`). Optionally also writes a CSV copy.
    """
    if csv_path:
        save_records_to_csv(records, csv_path)

    texts = [r["paragraph"] for r in records]
    embeddings = embed_texts(texts, model_name=model_name)

    index = build_index(embeddings)
    save_kb(index, records, faiss_path, metadata_path)

    print(f"Built knowledge base with {len(records)} entries.")


def append_to_knowledge_base(
    new_records,
    faiss_path,
    metadata_path,
    model_name="all-MiniLM-L6-v2",
):
    """Append new records to an existing knowledge base."""
    index, metadata = load_kb(faiss_path, metadata_path)

    texts = [r["paragraph"] for r in new_records]
    embeddings = embed_texts(texts, model_name=model_name)

    index.add(embeddings)
    metadata.extend(new_records)
    save_kb(index, metadata, faiss_path, metadata_path)

    print(f"Appended {len(new_records)} entries. KB now has {len(metadata)} total.")


def query_knowledge_base(
    query,
    faiss_path,
    metadata_path,
    model_name="all-MiniLM-L6-v2",
    top_k=5,
):
    """Search the knowledge base. Returns a list of records (dicts)."""
    index, metadata = load_kb(faiss_path, metadata_path)

    model = SentenceTransformer(model_name)
    query_vec = model.encode([query])

    distances, indices = index.search(np.array(query_vec), top_k)
    return [metadata[i] for i in indices[0]]