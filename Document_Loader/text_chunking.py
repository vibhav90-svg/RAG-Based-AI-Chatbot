
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# PDF file path
pdf_path = "data/POM_Teacher_Assessment_2_Apple.pdf"

# Load PDF
reader = PdfReader(pdf_path)

# Extract text from all pages
full_text = ""

for page in reader.pages:
    page_text = page.extract_text()

    if page_text:
        full_text += page_text + "\n"

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

# Split text into chunks
chunks = text_splitter.split_text(full_text)

# Display results
print("Text Chunking Completed Successfully!")
print("Total Pages:", len(reader.pages))
print("Total Chunks:", len(chunks))

# Display first chunk
if chunks:
    print("\nFirst Chunk:\n")
    print(chunks[0])
