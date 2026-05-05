"""
OpenAI's LLM calls and the RAG pipeline that combines retrieval with generation.

Set your API key via the OPENAI_API_KEY environment variable, or pass it
explicitly to `make_client()`.

Run `conda install openai` prior to using this code.
"""

import os
from typing import Optional

import openai

from logic import query_knowledge_base


DEFAULT_SYSTEM_PROMPT = (
    "Please tailor your response to be comprehensible, relevant, and empathetic."
)

# ---------------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------------

def make_client(api_key: Optional[str] = None) -> openai.OpenAI:
    """
    Create an OpenAI client. If `api_key` is None, reads from the
    OPENAI_API_KEY environment variable.
    """
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError(
            "No OpenAI API key found. Set OPENAI_API_KEY environment variable "
            "or pass api_key= explicitly."
        )
    return openai.OpenAI(api_key=key)


def call_llm(
    client: openai.OpenAI,
    query: str,
    context: str,
    model: str = "gpt-4o-2024-11-20",
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    temperature: float = 1e-5,
    max_tokens: int = 1000,
) -> str:
    """
    Send a query (with optional retrieved context) to the LLM and return
    the response text. If `context` is empty, only the query is sent.
    """
    if context:
        user_content = f"Context: {context}\nQuery: {query}"
    else:
        user_content = query

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()

# ---------------------------------------------------------------------------
# RAG pipeline
# ---------------------------------------------------------------------------

def format_context(paragraphs: list) -> str:
    """Format retrieved records into a numbered context block."""
    return "\n\n".join(
        f"{i}. {p['paragraph']}" for i, p in enumerate(paragraphs)
    )

def run_rag(
    query: str,
    faiss_path: str,
    metadata_path: str,
    client: openai.OpenAI,
    embedding_model: str = "all-MiniLM-L6-v2",
    llm_model: str = "gpt-4o-2024-11-20",
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    top_k: int = 5,
    include_context: bool = True,
) -> str:
    """
    Run the RAG pipeline.

    Retrieves the top-k most relevant paragraphs from the knowledge base,
    passes them as context to the LLM, and returns the formatted result.

    If `top_k` is 0, no retrieval happens — the query is sent to the LLM
    on its own (useful as a baseline).
    """
    if top_k == 0:
        paragraphs = []
        context = ""
    else:
        paragraphs = query_knowledge_base(
            query=query,
            faiss_path=faiss_path,
            metadata_path=metadata_path,
            model_name=embedding_model,
            top_k=top_k,
        )
        context = format_context(paragraphs)

    answer = call_llm(
        client=client,
        query=query,
        context=context,
        model=llm_model,
        system_prompt=system_prompt,
    )

    if include_context and context:
        return f"(Top-{top_k}) Query: {query}\nContext: {context}\nAnswer: {answer}"
    return f"(Top-{top_k}) Query: {query}\nAnswer: {answer}"