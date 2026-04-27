import pymupdf
import pymupdf4llm
import easyocr

print("Loading EasyOCR Model for Scanned PDFs...")
reader = easyocr.Reader(['en', 'fr'], gpu=False)

def process_pdf(file_path: str) -> str:
    """
    Attempts to read the PDF. If it is a digital PDF, uses PyMuPDF.
    If it is a scanned PDF (image-based), falls back to EasyOCR.
    """
    extracted_text = ""
    
    # ATTEMPT 1: The advanced layout extractor
    try:
        print("Extracting text using pymupdf4llm...")
        extracted_text = pymupdf4llm.to_markdown(file_path)
    except Exception as e:
        print(f"PyMuPDF4LLM Layout engine failed or bypassed: {str(e)[:100]}")
        
    # ATTEMPT 2: Fallback to standard PyMuPDF
    if not extracted_text or not extracted_text.strip():
        try:
            print("Falling back to standard PyMuPDF extraction...")
            doc = pymupdf.open(file_path)
            fallback_text = []
            for page in doc:
                fallback_text.append(page.get_text())
            extracted_text = "\n".join(fallback_text)
        except Exception as e:
            print(f"Standard extraction failed: {e}")

    # ATTEMPT 3: The PDF is a scanned image (No text layer found)
    if not extracted_text or not extracted_text.strip():
        print("No text layer found. This is a SCANNED PDF. Running EasyOCR...")
        try:
            doc = pymupdf.open(file_path)
            ocr_text_parts = []
            
            for page_num in range(len(doc)):
                print(f"Running OCR on page {page_num + 1}...")
                page = doc[page_num]
                
                # Convert the PDF page into a high-quality image
                pix = page.get_pixmap(dpi=200) 
                image_bytes = pix.tobytes("png")
                
                # Pass the raw image bytes to EasyOCR
                ocr_results = reader.readtext(image_bytes, detail=0)
                
                if ocr_results:
                    ocr_text_parts.extend(ocr_results)
                    
            extracted_text = "\n".join(ocr_text_parts)
            
        except Exception as e:
            print(f"OCR Phase failed: {e}")

    return extracted_text