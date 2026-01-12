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
    rewrite_style: str | None = None

# -------------------
# System prompt (LOCKED)
# -------------------
SYSTEM_PROMPT = """
You are a confident, socially intelligent communication assistant.

The user already knows WHAT they want to say.
Your job is ONLY to rewrite it with better word choice.

Rules:
- Sound natural, human, and confident
- Avoid lectures, explanations, or advice
- Prefer emotionally affirming language when appropriate
- Never judge or moralize
- Do NOT explain your thinking
- Do NOT add extra commentary

Return ONLY valid JSON in this exact format:
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

    # -------- Rewrite style instruction --------
    style_instruction = ""

    if req.rewrite_style == "Softer":
        style_instruction = "Rewrite the reply in a softer, warmer, emotionally gentle tone."

    elif req.rewrite_style == "More confident":
        style_instruction = "Rewrite the reply to sound confident, decisive, and attractive."

    elif req.rewrite_style == "More expressive":
        style_instruction = (
            "Rewrite the reply to be emotionally expressive and affirming. "
            "Add warmth and reassurance without sounding exaggerated, cheesy, or fake."
        )

    elif req.rewrite_style == "Shorter":
        style_instruction = "Rewrite the reply to be very concise while keeping warmth and clarity."

    # -------- User prompt --------
    user_prompt = f"""
Conversation:
{req.content}

Conversation type:
{req.conversation_type}

User goal:
{req.goal}

{style_instruction}
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
        return parsed

    except Exception:
        # Safe fallback (never break frontend)
        return {
            "summary": "The conversation needs a thoughtful, well-phrased response.",
            "risk": None,
            "best_reply": "That sounds good — I really appreciate you saying that.",
            "alternative_reply": "Honestly, that means a lot to hear.",
            "avoid_saying": "Anything dismissive, defensive, or overly formal."
        }
