from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

class ClinicalNote(BaseModel):
    text: str

@app.post("/predict")
async def predict_icd(note: ClinicalNote):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Act as a professional medical coder. Provide all possible ICD-10 codes."},
            {"role": "user", "content": note.text}
        ],
        max_tokens=300
    )

    return {
        "result": response.choices[0].message.content
    }
