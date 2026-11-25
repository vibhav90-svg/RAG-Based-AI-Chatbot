
import os
import re
from functools import lru_cache

from dotenv import load_dotenv
from fastembed import TextEmbedding
from openai import OpenAI
from qdrant_client import QdrantClient


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(PROJECT_DIR, ".env"))

collection_name = "rag_documents"


@lru_cache(maxsize=1)
def get_embedding_model():
    return TextEmbedding(model_name="BAAI/bge-small-en-v1.5")


@lru_cache(maxsize=1)
def get_vector_client():
    return QdrantClient(path=os.path.join(PROJECT_DIR, "qdrant_db"))


@lru_cache(maxsize=1)
def get_llm_client():
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError(
            "NVIDIA_API_KEY is missing. Add it to .env in the project folder."
        )
    return OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key,
    )


def retrieve_documents(question):
    query_vector = list(get_embedding_model().query_embed([question]))[0].tolist()
    client = get_vector_client()

    if not client.collection_exists(collection_name):
        raise RuntimeError(
            f"Qdrant collection '{collection_name}' was not found. "
            "Run Document_Loader/vector_store.py to index the PDF."
        )

    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=3,
        with_payload=True,
    )

    return "\n\n".join(
        text
        for result in results.points
        if (text := (result.payload or {}).get("text", "")).strip()
    )


def generate_answer(question, context):
    prompt = f"""
Answer the question using only the provided context.

If the answer is not available in the context,
say that the information was not found in the PDF.

Context:
{context}

Question:
{question}

Answer:
"""

    response = get_llm_client().chat.completions.create(
        model="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful PDF-based RAG assistant.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.2,
        max_tokens=1500,
        extra_body={
            "chat_template_kwargs": {
                "enable_thinking": False,
            }
        },
    )

    return response.choices[0].message.content


def answer_question(question):
    if re.fullmatch(r"\s*(?:hi|hello|hey|good\s+(?:morning|afternoon|evening))[\s.!]*", question, re.IGNORECASE):
        return "Hi! Ask me a question about the Apple PDF knowledge base."

    context = retrieve_documents(question)
    if not context:
        return "I could not find any relevant information in the PDF."
    return generate_answer(question, context)
