from fastapi import FastAPI, File, UploadFile
import re
import json
import shutil
import os
from ocr_utils import extract_text
from llm_utils import analyze_lease
from fairness_utils import calculate_fairness
from database import SessionLocal, engine
from models import LeaseAnalysis
from sqlalchemy.orm import Session
from vin_utils import decode_vin

# Create DB tables
from database import Base
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Lease Document Analyzer API")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
LAST_UPLOADED_FILE = None


def extract_vin_from_text(text: str):
    """
    Extract VIN even if OCR breaks it with spaces or newlines.
    """
    cleaned = text.replace(" ", "").replace("\n", "").upper()

    # VIN pattern: exactly 17 characters, no I O Q
    pattern = r"[A-HJ-NPR-Z0-9]{17}"
    match = re.search(pattern, cleaned)
    return match.group(0) if match else None


# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/")
def root():
    return {"message": "Lease Document Analyzer API is running"}

@app.post("/upload/")
async def upload_document(file: UploadFile = File(...)):
    global LAST_UPLOADED_FILE

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    LAST_UPLOADED_FILE = file.filename 

    return {
        "filename": file.filename,
        "status": "uploaded successfully"
    }

from fastapi import Depends
from sqlalchemy.orm import Session

@app.get("/analyze/")
def analyze_document(db: Session = Depends(get_db)):
    try:
        global LAST_UPLOADED_FILE

        if not LAST_UPLOADED_FILE:
            return {"error": "No file uploaded yet. Please upload a file first."}

        file_path = os.path.join(UPLOAD_DIR, LAST_UPLOADED_FILE)

        if not os.path.exists(file_path):
            return {"error": "Uploaded file not found on server."}

        # CHECK IF FILE ALREADY EXISTS IN DATABASE
        existing_record = db.query(LeaseAnalysis)\
                            .filter(LeaseAnalysis.filename == LAST_UPLOADED_FILE)\
                            .first()

        # OCR
        extracted_text = extract_text(file_path)

        # LLM Analysis
        raw_output = analyze_lease(extracted_text)
        cleaned = re.sub(r"```json|```", "", raw_output).strip()
        parsed_json = json.loads(cleaned)

       # Extract VIN from LLM output
        vehicle_section = parsed_json.get("vehicle_details", {})
        vin = vehicle_section.get("vehicle_id_number")

        # Fallback: Extract VIN directly from OCR text if LLM missed it
        if not vin:
            vin = extract_vin_from_text(extracted_text)

        # Validate VIN length (must be exactly 17)
        if vin and len(vin) != 17:
            print("⚠️ Extracted value is not a valid VIN, ignoring:", vin)
            vin = None


        vehicle_api_data = None

        if vin:
            try:
                vehicle_api_data = decode_vin(vin)
            except Exception as e:
                vehicle_api_data = {"error": f"VIN decoding failed: {str(e)}"}

        # IF SAME FILE EXISTS WITH SAME VIN → RETURN STORED DATA
        if existing_record and existing_record.vin == vin:
            return {
                "message": "Existing analysis with VIN data found",
                "record_id": existing_record.id,
                "filename": existing_record.filename,
                "analysis_result": existing_record.analysis_result,
                "fairness_analysis": existing_record.fairness_analysis,
                "vehicle_api_data": existing_record.vehicle_api_data
            }

        # Fairness / SLA Evaluation
        fairness_result = calculate_fairness(parsed_json)

        # STORE IN MYSQL
        record = LeaseAnalysis(
            filename=LAST_UPLOADED_FILE,
            analysis_result=parsed_json,
            fairness_analysis=fairness_result,
            vin=vin,
            vehicle_api_data=vehicle_api_data
        )

        db.add(record)
        db.commit()
        db.refresh(record)

        return {
            "message": "SLA analysis completed and stored in database",
            "record_id": record.id,
            "filename": LAST_UPLOADED_FILE,
            "analysis_result": parsed_json,
            "fairness_analysis": fairness_result,
            "vehicle_api_data": vehicle_api_data
        }

    except Exception as e:
        return {
            "error": "Processing failed",
            "details": str(e)
        }
