
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi

# Load PDF
reader = PdfReader("data/POM_Teacher_Assessment_2_Apple.pdf")

full_text = ""

for page in reader.pages:
    text = page.extract_text()
    if text:
        full_text += text + "\n"

# Create chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_text(full_text)

# Initialize BM25
tokenized_chunks = [chunk.lower().split() for chunk in chunks]
bm25 = BM25Okapi(tokenized_chunks)

# Load embedding model
embedding_model = TextEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

# Connect to Qdrant
client = QdrantClient(path="./qdrant_db")

# Take user question
question = input("Ask your question: ")

# BM25 Search
tokenized_query = question.lower().split()
bm25_scores = bm25.get_scores(tokenized_query)

bm25_results = sorted(
    enumerate(bm25_scores),
    key=lambda x: x[1],
    reverse=True
)[:3]

# Vector Search
query_vector = list(
    embedding_model.query_embed([question])
)[0].tolist()

vector_results = client.query_points(
    collection_name="rag_documents",
    query=query_vector,
    limit=3,
    with_payload=True
)

# Combine results using Reciprocal Rank Fusion
combined_scores = {}

for rank, (index, score) in enumerate(bm25_results, start=1):
    combined_scores[index] = combined_scores.get(index, 0) + 1 / (60 + rank)

for rank, result in enumerate(vector_results.points, start=1):
    index = result.payload["chunk_id"]
    combined_scores[index] = combined_scores.get(index, 0) + 1 / (60 + rank)

# Display hybrid results
print("\nHybrid Search Results:\n")

for index, score in sorted(
    combined_scores.items(),
    key=lambda x: x[1],
    reverse=True
)[:3]:

    print("Chunk ID:", index)
    print("Combined Score:", round(score, 5))
    print("Content:", chunks[index])
    print("-" * 60)

client.close()
