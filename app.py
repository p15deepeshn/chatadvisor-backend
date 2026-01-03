from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from openai import OpenAI

# -------------------
# App initialization
# -------------------
app = FastAPI()

# -------------------
# CORS (MUST be here)
# -------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # MVP ONLY
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------
# OpenAI client
# -------------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -------------------
# Request schema
# -------------------
class AnalyzeRequest(BaseModel):
    content: str
    conversation_type: str
    goal: str

# -------------------
# Routes
# -------------------
@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    prompt = f"""
Conversation:
{req.content}

Type: {req.conversation_type}
Goal: {req.goal}

Give:
1. Summary
2. Risk (or null)
3. Best reply
4. Alternative reply
5. Avoid saying
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
    )

    text = response.choices[0].message.content

    return {
        "summary": text.split("\n")[0],
        "risk": None,
        "best_reply": text,
        "alternative_reply": text,
        "avoid_saying": "Avoid sounding pushy."
    }
