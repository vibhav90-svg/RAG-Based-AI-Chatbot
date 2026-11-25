
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models

# PDF file path
pdf_path = "data/POM_Teacher_Assessment_2_Apple.pdf"

# Load PDF
reader = PdfReader(pdf_path)

# Extract text
full_text = ""

for page in reader.pages:
    page_text = page.extract_text()

    if page_text:
        full_text += page_text + "\n"

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_text(full_text)

print("Total Chunks:", len(chunks))

# Load embedding model
embedding_model = TextEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

# Generate embeddings
embeddings = list(embedding_model.embed(chunks))

print("Embeddings Generated:", len(embeddings))

# Initialize Qdrant local database
client = QdrantClient(path="./qdrant_db")

collection_name = "rag_documents"

# Create collection if it does not exist
if not client.collection_exists(collection_name):
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=384,
            distance=models.Distance.COSINE
        )
    )

# Store vectors and original text
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

# Insert data into Qdrant
if points:
    client.upsert(
        collection_name=collection_name,
        points=points
    )

# Display database information
info = client.get_collection(collection_name)

print("\nQdrant Database Created Successfully!")
print("Collection Name:", collection_name)
print("Stored Vectors:", info.points_count)

# Close database connection
client.close()
