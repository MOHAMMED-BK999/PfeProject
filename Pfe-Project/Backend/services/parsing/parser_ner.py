import json
import torch
from typing import List, Dict, Any
from pydantic import BaseModel
from gliner import GLiNER
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from PIL import Image

# --- 1. STRICT PYDANTIC SCHEMAS ---

class PersonalInfo(BaseModel):
    name: str = ""
    email: str = ""
    phone: str = ""
    location: str = ""
    linkedin: str = ""

class Education(BaseModel):
    degree: str = ""
    institution: str = ""
    year: str = ""
    gpa: str = ""

class Experience(BaseModel):
    company: str = ""
    role: str = ""
    start: str = ""
    end: str = ""
    description: str = ""
    achievements: List[str] = []

class Skills(BaseModel):
    technical: List[str] = []
    soft: List[str] = []
    languages: List[str] = []

class Metadata(BaseModel):
    years_exp: str = ""
    last_role: str = ""
    education_level: str = ""

class TargetCVSchema(BaseModel):
    personal_info: PersonalInfo = PersonalInfo()
    education: List[Education] = []
    experience: List[Experience] = []
    skills: Skills = Skills()
    certifications: List[str] = []
    summary: str = ""
    metadata: Metadata = Metadata()

# --- 2. LOAD PRE-TRAINED COMMUNITY MODELS ---

print("Loading Model 1: GLiNER (Multi-lingual text processing)...")
gliner_model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")

print("Loading Model 2: LayoutLMv3 (Pre-trained CV visual layout)...")
processor = LayoutLMv3Processor.from_pretrained("Kiruba11/layoutlmv3-resume-ner2", apply_ocr=False)
layout_model = LayoutLMv3ForTokenClassification.from_pretrained("Kiruba11/layoutlmv3-resume-ner2")

# --- 3. EXTRACTION FUNCTIONS ---

def extract_with_gliner(text: str) -> Dict[str, List[str]]:
    """
    GLiNER handles text-based entities across multiple languages.
    """
    labels = [
        "Person Name", "Email Address", "Phone Number", "Location", "LinkedIn URL",
        "Degree", "University", "Graduation Year", "GPA",
        "Company", "Job Title", "Date", "Technical Skill", "Soft Skill", 
        "Language", "Certification"
    ]
    
    entities = gliner_model.predict_entities(text, labels, threshold=0.4)
    
    raw_data = {label: [] for label in labels}
    for ent in entities:
        clean_text = ent["text"].strip()
        if clean_text not in raw_data[ent["label"]]:
            raw_data[ent["label"]].append(clean_text)
            
    return raw_data

def extract_with_layoutlmv3(image_path: str, words: List[str], boxes: List[List[int]]) -> Dict[str, List[str]]:
    """
    Dynamically decodes LayoutLMv3 tokens using the model's internal ID mapping.
    """
    try:
        image = Image.open(image_path).convert("RGB")
        encoding = processor(image, words, boxes=boxes, return_tensors="pt", truncation=True, padding="max_length")
        
        with torch.no_grad():
            outputs = layout_model(**encoding)
            
        predictions = outputs.logits.argmax(-1).squeeze().tolist()
        id2label = layout_model.config.id2label
        
        extracted_entities = {}
        current_entity = ""
        current_label = None

        # Reconstruct words from BIO tags (B- = Beginning, I- = Inside)
        for word, pred in zip(words, predictions[:len(words)]):
            label = id2label.get(pred, "O")
            
            if label == "O":
                if current_entity and current_label:
                    extracted_entities.setdefault(current_label, []).append(current_entity.strip())
                    current_entity = ""
                    current_label = None
                continue
            
            clean_label = label.replace("B-", "").replace("I-", "")
            
            if label.startswith("B-"):
                if current_entity and current_label:
                    extracted_entities.setdefault(current_label, []).append(current_entity.strip())
                current_entity = word
                current_label = clean_label
            elif label.startswith("I-") and current_label == clean_label:
                current_entity += " " + word
                
        # Catch the last entity in the loop
        if current_entity and current_label:
            extracted_entities.setdefault(current_label, []).append(current_entity.strip())

        return extracted_entities
    except Exception as e:
        print(f"LayoutLMv3 processing skipped or failed: {e}")
        return {}

# --- 4. MASTER HYBRID PIPELINE ---

def process_cv_hybrid(text: str, image_path: str = None, words: List[str] = None, boxes: List[List[int]] = None) -> str:
    """
    Merges GLiNER text data and LayoutLMv3 visual data into a strict JSON schema.
    """
    print("\nRunning Extraction Pipeline with Pre-Trained Models...")
    
    cv_data = TargetCVSchema()
    
    # 1. Run GLiNER (Text Base)
    gliner_data = extract_with_gliner(text)
    
    # 2. Run LayoutLMv3 (Visual Base - Optional based on passed inputs)
    layout_data = {}
    if image_path and words and boxes:
        layout_data = extract_with_layoutlmv3(image_path, words, boxes)
        print(f"LayoutLMv3 found the following entity categories: {list(layout_data.keys())}")

    # --- MAP PERSONAL INFO (Prioritizing GLiNER for text specifics) ---
    if gliner_data["Person Name"]: cv_data.personal_info.name = gliner_data["Person Name"][0]
    if gliner_data["Email Address"]: cv_data.personal_info.email = gliner_data["Email Address"][0]
    if gliner_data["Phone Number"]: cv_data.personal_info.phone = gliner_data["Phone Number"][0]
    if gliner_data["Location"]: cv_data.personal_info.location = gliner_data["Location"][0]
    if gliner_data["LinkedIn URL"]: cv_data.personal_info.linkedin = gliner_data["LinkedIn URL"][0]

    # --- MAP SKILLS & CERTIFICATIONS ---
    cv_data.skills.technical = list(set(gliner_data["Technical Skill"]))
    cv_data.skills.soft = list(set(gliner_data["Soft Skill"]))
    cv_data.skills.languages = list(set(gliner_data["Language"]))
    cv_data.certifications = list(set(gliner_data["Certification"]))

    # --- MAP EDUCATION ---
    degrees = gliner_data["Degree"]
    unis = gliner_data["University"]
    years = gliner_data["Graduation Year"]
    gpas = gliner_data["GPA"]
    
    max_edu = max(len(degrees), len(unis))
    for i in range(max_edu):
        cv_data.education.append(Education(
            degree=degrees[i] if i < len(degrees) else "",
            institution=unis[i] if i < len(unis) else "",
            year=years[i] if i < len(years) else "",
            gpa=gpas[i] if i < len(gpas) else ""
        ))

    # --- MAP EXPERIENCE ---
    companies = gliner_data["Company"]
    roles = gliner_data["Job Title"]
    dates = gliner_data["Date"]
    
    max_exp = max(len(companies), len(roles))
    for i in range(max_exp):
        cv_data.experience.append(Experience(
            company=companies[i] if i < len(companies) else "",
            role=roles[i] if i < len(roles) else "",
            start=dates[i] if i < len(dates) else "", 
        ))

    # --- MAP METADATA ---
    if cv_data.experience:
        cv_data.metadata.last_role = cv_data.experience[0].role
    if cv_data.education:
        cv_data.metadata.education_level = cv_data.education[0].degree

    # Optional: If LayoutLMv3 finds things GLiNER missed, we could merge them here
    # Since we don't know exact LayoutLMv3 label strings yet (e.g., 'SKILL' vs 'Skill'), 
    # we are keeping it safe by printing them out in the terminal first.

    return json.dumps(cv_data.model_dump(), indent=4, ensure_ascii=False)

if __name__ == "__main__":
    sample_text = """
    JEAN DUPONT
    Lyon, France | jean.dupont@email.com | +33 6 98 76 54 32
    
    EDUCATION
    Licence en Ingénierie Financière, Université de Lyon, 2020
    
    EXPERIENCE
    Financial Analyst - BNP Paribas (2021 - Present)
    
    SKILLS
    Excel, Financial Modeling, Communication, French, Arabic
    """
    
    # Passing None for visual data to trigger standard text pipeline testing
    final_json = process_cv_hybrid(text=sample_text, image_path=None, words=None, boxes=None)
    
    print("\n--- FINAL STRUCTURED JSON OUTPUT ---")
    print(final_json)