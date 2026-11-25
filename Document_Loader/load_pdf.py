
from pypdf import PdfReader

# PDF file path
pdf_path = "data/POM_Teacher_Assessment_2_Apple.pdf"

# Read PDF file
reader = PdfReader(pdf_path)

# Extract pages
print("PDF Loaded Successfully!")
print("Total Pages:", len(reader.pages))

# Extract first page text
if len(reader.pages) > 0:

    first_page = reader.pages[0]

    text = first_page.extract_text()

    print("\nFirst Page Content:\n")
    print(text[:1000])

    print("\nPage Metadata:")
    print(reader.metadata)