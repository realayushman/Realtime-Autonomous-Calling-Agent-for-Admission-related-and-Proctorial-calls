from langchain_community.document_loaders import PyPDFLoader

pdf_path = "data/MIT-Facts2024-Accessible-with-Map-and-Cover.pdf"

loader = PyPDFLoader(pdf_path)
docs = loader.load()

print(f"Number of pages loaded: {len(docs)}")
for i, doc in enumerate(docs):
    print(f"\n--- Page {i+1} preview ---")
    print(doc.page_content[:300])  # prints first 300 characters of each page
