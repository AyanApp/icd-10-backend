from fastapi import FastAPI
from pydantic import BaseModel
import os, json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

class ClinicalNote(BaseModel):
    text: str

@app.post("/predict")
async def predict_icd(note: ClinicalNote):
    system_message = """
    You are a medical coder. Extract all ICD-10 codes from the note.
    Output ONLY in this JSON format:
    [
      {"code": "ICD10_CODE", "description": "Meaning"}
    ]
    No extra text. No explanation.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": note.text}
        ],
        max_tokens=300,
        temperature=0
    )

    raw = response.choices[0].message.content

    # Convert model text → real JSON
    try:
        parsed = json.loads(raw)
    except:
        cleaned = raw.strip().replace("```json", "").replace("```", "")
        parsed = json.loads(cleaned)

    return parsed
