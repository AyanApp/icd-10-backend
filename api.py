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

@app.post("/predict")
async def predict_icd(note: ClinicalNote):
    delimiter = "####"

    system_message = f"""
    Act as a professional medical coder.
    Provide all possible ICD-10 codes based on the clinical notes.
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
