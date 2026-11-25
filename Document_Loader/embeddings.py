
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from fastembed import TextEmbedding

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

# Display results
print("\nEmbeddings Generated Successfully!")

print("Total Embeddings:", len(embeddings))

if embeddings:
    print("Vector Dimensions:", len(embeddings[0]))
    print("\nFirst Embedding:")
    print(embeddings[0][:10])
