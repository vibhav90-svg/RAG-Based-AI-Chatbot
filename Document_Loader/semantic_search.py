
from fastembed import TextEmbedding
from qdrant_client import QdrantClient

# Initialize embedding model
embedding_model = TextEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

# Connect to local Qdrant database
client = QdrantClient(path="./qdrant_db")

collection_name = "rag_documents"

# Take question from user
question = input("Ask a question about your PDF: ")

# Convert question into embedding
query_embedding = list(
    embedding_model.query_embed([question])
)[0].tolist()

# Search relevant information
results = client.query_points(
    collection_name=collection_name,
    query=query_embedding,
    limit=3,
    with_payload=True
)

# Display search results
print("\nRelevant Information Found:\n")

for i, result in enumerate(results.points, start=1):
    print(f"Result {i}")
    print("Similarity Score:", round(result.score, 4))
    print("Content:", result.payload["text"])
    print("-" * 60)

# Close database connection
client.close()
