import os
import pdfplumber
import pymupdf4llm
import unicodedata

def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize('NFKC', text)
    text = text.encode('utf-8', 'ignore').decode('utf-8')
    return text.strip()

def is_native_pdf(file_path: str) -> bool:
    try:
        with pdfplumber.open(file_path) as pdf:
            text = ""
            if len(pdf.pages) > 0:
                text = pdf.pages[0].extract_text() or ""
            return len(text.strip()) > 50
    except Exception as e:
        print(f"Error checking PDF type: {e}")
        return False

def extract_native_pdf(file_path: str) -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    print(f"Processing: {file_path}")

    if not is_native_pdf(file_path):
        print("Warning: This PDF appears to be a scanned image.")
        print("This script is designed for Native PDFs. The output may be empty.")

    print("Extracting text using pymupdf4llm...")
    raw_text = pymupdf4llm.to_markdown(file_path)

    print("Normalizing text...")
    cleaned_text = normalize_text(raw_text)

    return cleaned_text

if __name__ == "__main__":
    test_pdf_path = "C:\\Users\\DELL\\OneDrive\\Desktop\\2025-2026.pdf"

    if not os.path.exists(test_pdf_path):
        print(f"Please change 'test_pdf_path' to point to a real PDF on your computer.")
    else:
        extracted_content = extract_native_pdf(test_pdf_path)

        print("\n--- EXTRACTED TEXT ---")
        print(extracted_content)
        print("----------------------")