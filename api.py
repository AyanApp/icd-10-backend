from fastapi import FastAPI
from pydantic import BaseModel
import openai
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

class ClinicalNote(BaseModel):
    text: str

class SuggestRequest(BaseModel):
    note: str
    codes: list  # Example: ["R07.9 Chest pain", "R06.02 Shortness of Breath"]


# =====================================================
# 1️⃣ MAIN ICD PREDICTION (LEVEL 1, LEVEL 2, LEVEL 3)
# =====================================================
@app.post("/predict")
async def predict_icd(note: ClinicalNote):
    delimiter = "####"

    system_message = """
    Act as a professional medical coder.
    Provide ALL possible ICD-10 codes from the clinical notes.
    Only return valid ICD-10 codes with descriptions.
    """

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"{delimiter}{note.text}{delimiter}"}
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=500
    )

    return {"result": response.choices[0].message["content"]}


# =====================================================
# 2️⃣ AUTO-SUGGEST NEXT DIAGNOSIS (LEVEL 2)
# =====================================================
@app.post("/suggest_next")
async def suggest_next(payload: SuggestRequest):
    prompt = f"""
    You are a senior physician.

    Based on:
    - Clinical Notes: {payload.note}
    - ICD-10 Codes Found: {payload.codes}

    Suggest what the DOCTOR should evaluate NEXT.
    Provide ONLY one short, clear recommended next-level diagnosis or assessment.
    """

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150
    )

    suggestion = response.choices[0].message["content"]
    return {"suggestion": suggestion}


# =====================================================
# 3️⃣ OPTIONAL: AUTO-SUGGEST NEXT DIAGNOSIS (LEVEL 3)
# =====================================================
@app.post("/suggest_next_level3")
async def suggest_next_level3(payload: SuggestRequest):
    prompt = f"""
    Based on previous levels:
    - Notes: {payload.note}
    - ICD Codes: {payload.codes}

    Suggest the next most logical diagnostic step (Level 3).
    Keep the output short (1–2 lines) and clinically relevant.
    """

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150
    )

    suggestion = response.choices[0].message["content"]
    return {"suggestion": suggestion}
