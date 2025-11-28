from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import os

app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Allow Flutter / Web
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Helper function to call OpenAI
# ----------------------------
async def ask_openai(prompt: str):
    response = client.chat.completions.create(
        model="gpt-4o-mini",   # ★ cheapest but accurate
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300
    )
    return response.choices[0].message["content"]


# ----------------------------
# Level 1 Prediction
# ----------------------------
@app.post("/level1")
async def level1(request: Request):
    body = await request.json()
    notes = body.get("clinical_notes", "")

    prompt = f"""
    Based on this clinical note provide the TOP 5 most likely Level-1 ICD categories.

    Clinical notes: {notes}

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


# ----------------------------
# Level 2 Prediction
# ----------------------------
@app.post("/level2")
async def level2(request: Request):
    body = await request.json()
    level1_selected = body.get("selected_level1", "")
    notes = body.get("clinical_notes", "")

    prompt = f"""
    The doctor selected this Level-1 ICD category:
    {level1_selected}

    Based on clinical notes below, suggest the TOP 5 most likely Level-2 ICD codes.

    Clinical notes: {notes}

    Output JSON with best match first.
    """

    answer = await ask_openai(prompt)
    return {"result": answer}


# ----------------------------
# Level 3 Prediction
# ----------------------------
@app.post("/level3")
async def level3(request: Request):
    body = await request.json()
    level2_selected = body.get("selected_level2", "")
    notes = body.get("clinical_notes", "")

    prompt = f"""
    The doctor selected this Level-2 ICD code:
    {level2_selected}

    Based on the clinical notes, give TOP 5 most accurate Level-3 ICD codes.

    Clinical notes: {notes}

    Output JSON.
    """

    answer = await ask_openai(prompt)
    return {"result": answer}


# ----------------------------
# Direct ICD Prediction (optional)
# ----------------------------
@app.post("/predict_icd")
async def predict_icd(request: Request):
    body = await request.json()
    notes = body.get("clinical_notes", "")

    prompt = f"""
    You are an ICD-10 coding engine.
    Based on the clinical notes, return the TOP 5 most accurate ICD-10 codes.

    Clinical notes: {notes}

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
