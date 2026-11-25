
import os
import re
from functools import lru_cache

import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from fastembed import TextEmbedding
from openai import APIStatusError, OpenAI
from qdrant_client import QdrantClient, models


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(
    PROJECT_DIR, "data", "POM_Teacher_Assessment_2_Apple.pdf"
)

load_dotenv(os.path.join(PROJECT_DIR, ".env"))

COLLECTION_NAME = "rag_documents"


@lru_cache(maxsize=1)
def get_embedding_model():
    return TextEmbedding(model_name="BAAI/bge-small-en-v1.5")


@lru_cache(maxsize=1)
def get_vector_client():
    db_path = os.path.join(PROJECT_DIR, "qdrant_db")
    os.makedirs(db_path, exist_ok=True)
    return QdrantClient(path=db_path)


@lru_cache(maxsize=1)
def get_llm_client():
    api_key = os.getenv("NVIDIA_API_KEY")

    if not api_key:
        try:
            api_key = st.secrets["NVIDIA_API_KEY"]
        except Exception:
            api_key = None

    if not api_key:
        raise RuntimeError(
            "NVIDIA_API_KEY is missing. Add it to Streamlit Secrets."
        )

    return OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key,
    )


def create_knowledge_base():
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(
            f"PDF file not found: {PDF_PATH}"
        )

    reader = PdfReader(PDF_PATH)

    full_text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            full_text += page_text + "\n"

    if not full_text.strip():
        raise RuntimeError("No readable text found in the PDF.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_text(full_text)

    if not chunks:
        raise RuntimeError("No text chunks were generated.")

    embedding_model = get_embedding_model()
    embeddings = list(embedding_model.embed(chunks))

    client = get_vector_client()

    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=384,
            distance=models.Distance.COSINE
        )
    )

    points = []

    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        points.append(
            models.PointStruct(
                id=i,
                vector=embedding.tolist(),
                payload={
                    "text": chunk,
                    "chunk_id": i
                }
            )
        )

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )

    return len(chunks)


def ensure_knowledge_base():
    client = get_vector_client()

    if not client.collection_exists(COLLECTION_NAME):
        create_knowledge_base()
        return

    info = client.get_collection(COLLECTION_NAME)

    if info.points_count == 0:
        create_knowledge_base()


def retrieve_documents(question):
    ensure_knowledge_base()

    query_vector = list(
        get_embedding_model().query_embed([question])
    )[0].tolist()

    client = get_vector_client()

    results = client.query_points(
        collection_name=COLLECTION_NAME,
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

    try:
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

    except APIStatusError as exc:
        detail = exc.body

        if isinstance(detail, dict):
            detail = (
                detail.get("detail")
                or detail.get("message")
                or detail.get("error")
                or detail
            )

        raise RuntimeError(
            f"NVIDIA API request failed (HTTP {exc.status_code}): {detail}"
        ) from exc

    return response.choices[0].message.content


def answer_question(question):
    if re.fullmatch(
        r"\s*(?:hi|hello|hey|good\s+(?:morning|afternoon|evening))[\s.!]*",
        question,
        re.IGNORECASE
    ):
        return "Hi! Ask me a question about the Apple PDF knowledge base."

    context = retrieve_documents(question)

    if not context:
        return "I could not find any relevant information in the PDF."

    return generate_answer(question, context)
