from fastapi import FastAPI
from pydantic import BaseModel
import os
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
    You are a professional medical coder.
    Extract ALL possible ICD-10 codes from the clinical note.
    Output STRICTLY in JSON array format:
    [
      {"code": "ICD10_CODE", "description": "Meaning"}
    ]
    No extra text. No explanation. JSON only.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": note.text}
        ],
        max_tokens=500,
        temperature=0
    )

    result = response.choices[0].message.content
    return {"result": result}
