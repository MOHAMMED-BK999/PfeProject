import os

# --- IMPORT PHASE 1: INGESTION SCRIPTS ---
from services.extraction.extractor_pdf import process_pdf 
from services.extraction.extractor_word import extract_all_from_docx

# --- IMPORT PHASE 2: NER EXTRACTION ---
# Importing the updated 2-model pipeline (LayoutLMv3 + GLiNER)
from services.parsing.parser_ner import process_cv_hybrid

def extract_text_based_on_filetype(file_path: str) -> str:
    """
    ROUTER FUNCTION: Checks the file extension and uses the correct extractor.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return ""

    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension == ".pdf":
        print("Detected PDF document. Routing to PDF Extractor...")
        return process_pdf(file_path)
        
    elif file_extension == ".docx":
        print("Detected Word document. Routing to DOCX Extractor...")
        return extract_all_from_docx(file_path)
        
    else:
        print(f"Warning: Unsupported file type: {file_extension}. Please use .pdf or .docx")
        return ""

def run_end_to_end_pipeline(file_path: str):
    print(f"\n{'='*50}")
    print(f"STARTING PIPELINE FOR: {os.path.basename(file_path)}")
    print(f"{'='*50}")

    # STEP 1: Ingestion (Auto-Routed)
    print("\n[Step 1] Reading File & Extracting Text...")
    raw_text = extract_text_based_on_filetype(file_path)
    
    if not raw_text or not raw_text.strip():
        print("Error: Failed to extract any text. Aborting pipeline.")
        return None
        
    print("\nText extracted successfully.")
    print("--- PREVIEW OF RAW TEXT ---")
    print(raw_text[:300] + "...\n[TRUNCATED]") 

    # STEP 2: Extraction NLP (LayoutLMv3 + GLiNER)
    print("\n[Step 2] Extracting Entities via 2-Model NLP Pipeline...")
    
    extracted_entities_json = process_cv_hybrid(
        text=raw_text, 
        image_path=None, 
        words=None, 
        boxes=None
    )

    print("\nPIPELINE COMPLETE.")
    return extracted_entities_json

if __name__ == "__main__":
    # Ensure this path points to a valid PDF or DOCX file on your machine
    test_file = r"C:\\Users\\DELL\\OneDrive\\Documents\\archive\\data\\data\\INFORMATION-TECHNOLOGY\\info1.pdf"
    
    final_data = run_end_to_end_pipeline(test_file)
    
    if final_data:
        print("\n--- FINAL STRUCTURED JSON OUTPUT ---")
        print(final_data)