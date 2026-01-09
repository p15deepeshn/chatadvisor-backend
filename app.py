from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import json
from openai import OpenAI

# -------------------
# App initialization
# -------------------
app = FastAPI()

# -------------------
# CORS (MVP SAFE)
# -------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # MVP only
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
# System prompt (CRITICAL)
# -------------------
SYSTEM_PROMPT = """
You are a confident, socially intelligent communication assistant.

The user already knows what they want to say.
Your job is NOT to give advice or moral lessons.
Your job is to rewrite the reply using better word choice.

Guidelines:
- Sound natural, confident, and human
- Avoid being preachy, philosophical, or overly safe
- Prefer emotionally affirming language when appropriate
- Never shame or judge the user
- Do not explain — just write the reply

Conversation:
{conversation}

Context:
{conversation_type}

User goal:
{goal}

If a rewrite style is provided, apply it strictly:
- Softer → reduce intensity, keep warmth
- More confident → assertive, attractive, clear
- Romantic → affirming, emotionally warm (not creepy)
- Playful → light, teasing, natural
- Shorter → concise, no filler

Return exactly this JSON:
{
  "summary": "1–2 lines explaining what’s happening",
  "risk": "one short line OR null",
  "best_reply": "best phrased reply",
  "alternative_reply": "another phrasing with a different tone",
  "avoid_saying": "one example of wording to avoid"
}
"""

# -------------------
# Routes
# -------------------
@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    user_prompt = f"""
Conversation:
{req.content}

Conversation type: {req.conversation_type}
Goal: {req.goal}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.4,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    raw = response.choices[0].message.content.strip()

    try:
        parsed = json.loads(raw)
    except Exception:
        return {
            "summary": "Unable to analyze conversation clearly.",
            "risk": "",
            "best_reply": "Thanks for reaching out. I’ll get back to you later.",
            "alternative_reply": "Appreciate the message. I’ll review this and respond.",
            "avoid_saying": "Anything confrontational or dismissive."
        }

    return parsed
