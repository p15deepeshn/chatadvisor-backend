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
You are ChatAdvisor, a reply-writing assistant.

STRICT RULES:
- Respond ONLY in valid JSON.
- Do NOT include explanations inside replies.
- Do NOT repeat summary or risk inside replies.
- Do NOT use numbering, markdown, labels, or emojis inside values.
- Replies must be ready to copy and send.

Return EXACTLY this JSON shape:

{
  "summary": "Brief 1–2 line explanation of what is happening",
  "risk": "Potential risk if any, otherwise empty string",
  "best_reply": "One clear reply message",
  "alternative_reply": "A different valid reply message",
  "avoid_saying": "What the user should avoid saying"
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
