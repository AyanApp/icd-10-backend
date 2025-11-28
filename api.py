from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import os

app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ------------------------
# CORS (for Flutter / web)
# ------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------
# Helper function to call OpenAI
# ------------------------
async def ask_openai(prompt: str):
    """
    Calls OpenAI chat completions and returns the combined content from all choices.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",   # cheapest accurate model
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300
    )

    # New SDK: use .content, not subscript
    return " ".join([choice.message.content for choice in response.choices])


# ------------------------
# Level 1 Prediction
# ------------------------
@app.post("/level1")
async def level1(request: Request):
    body = await request.json()
    clinical_notes = body.get("clinical_notes", "")

    prompt = f"""
    Based on this clinical note provide the TOP 5 most likely Level-1 ICD categories.

    Clinical notes: {clinical_notes}

    Output JSON:
    {{
      "level1": [
        {{"code": "XXX", "description": "..." }},
        ...
      ]
    }}
    """

    answer = await ask_openai(prompt)
    return {"result": answer}


# ------------------------
# Level 2 Prediction
# ------------------------
@app.post("/level2")
async def level2(request: Request):
    body = await request.json()
    selected_level1 = body.get("selected_level1", "")
    clinical_notes = body.get("clinical_notes", "")

    prompt = f"""
    The doctor selected this Level-1 ICD category:
    {selected_level1}

    Based on clinical notes below, suggest the TOP 5 most likely Level-2 ICD codes.

    Clinical notes: {clinical_notes}

    Output JSON with best match first.
    """

    answer = await ask_openai(prompt)
    return {"result": answer}


# ------------------------
# Level 3 Prediction
# ------------------------
@app.post("/level3")
async def level3(request: Request):
    body = await request.json()
    selected_level2 = body.get("selected_level2", "")
    clinical_notes = body.get("clinical_notes", "")

    prompt = f"""
    The doctor selected this Level-2 ICD code:
    {selected_level2}

    Based on the clinical notes, give TOP 5 most accurate Level-3 ICD codes.

    Clinical notes: {clinical_notes}

    Output JSON.
    """

    answer = await ask_openai(prompt)
    return {"result": answer}


# ------------------------
# Direct ICD Prediction
# ------------------------
@app.post("/predict_icd")
async def predict_icd(request: Request):
    body = await request.json()
    clinical_notes = body.get("clinical_notes", "")

    prompt = f"""
    You are an ICD-10 coding engine.
    Based on the clinical notes, return the TOP 5 most accurate ICD-10 codes.

    Clinical notes: {clinical_notes}

    Output JSON:
    {{
      "icd": [
        {{"code": "R07.9", "description": "Chest pain, unspecified"}},
        ...
      ]
    }}
    """

    answer = await ask_openai(prompt)
    return {"result": answer}
