import fitz  # PyMuPDF

pdf_path = "data/MIT-Facts2024-Accessible-with-Map-and-Cover.pdf"

doc = fitz.open(pdf_path)
print(f"Total pages: {doc.page_count}")

for i, page in enumerate(doc):
    text = page.get_text()
    print(f"\n--- Page {i+1} preview ---")
    print(text[:300])
