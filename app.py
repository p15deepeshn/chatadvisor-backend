from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI, OpenAIError, RateLimitError
import os

# =====================
# CONFIG
# =====================
USE_AI = True  # Set False to force mock mode

# =====================
# APP SETUP
# =====================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================
# SCHEMA
# =====================
class AnalyzeRequest(BaseModel):
    content: str
    conversation_type: str
    goal: str
    rewrite_style: str | None = None


# =====================
# MOCK RESPONSE (fallback)
# =====================
def mock_response():
    return {
        "summary": "She’s responding politely but with low emotional engagement.",
        "risk": "Pushing for emotional clarity right now may create pressure.",
        "best_reply": "Got it — sounds like a long day. We can catch up later.",
        "alternative_reply": "No worries. Hope you get some rest.",
        "avoid_saying": "Avoid turning this into a serious conversation right now."
    }

# =====================
# OPENAI SETUP
# =====================
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """
You are ChatAdvisor — a calm, socially intelligent human advisor.

Your job is to help the user decide what to say NEXT in a real chat.

Core principles:
- Be practical, not philosophical
- Sound like a thoughtful human, not a therapist
- Replies must be short, natural, and actually sendable
- Match the other person's energy (do NOT escalate)
- If interest is low or replies are dry, reduce intensity
- Never guilt, pressure, or manipulate
- Avoid clichés and generic advice

Tone rules:
- Dating / personal: warm, casual, confident
- Professional: clear, respectful, concise
- Conflict: de-escalating, composed
- Sales: polite, low-pressure

Output rules:
- NO emojis unless tone clearly allows it
- NO moral lectures
- NO multiple paragraphs
- NO analysis inside replies

Respond ONLY in valid JSON with this structure:
{
  "summary": "1-sentence neutral summary",
  "risk": "Brief risk or empty string",
  "best_reply": "Best next message to send",
  "alternative_reply": "Lower-pressure option",
  "avoid_saying": "One thing to avoid saying"
}
"""

# =====================
# ENDPOINT
# =====================
@app.post("/analyze")
def analyze(req: AnalyzeRequest):

    # ---------- Interest sensitivity ----------
    low_interest_hint = ""
    if any(x in req.content.lower() for x in ["ok", "busy", "later", "seen", "hmm"]):
        low_interest_hint = (
            "The other person seems low on engagement. "
            "Suggest low-pressure, non-needy replies."
        )

    # ---------- Mock mode ----------
    if not USE_AI:
        return mock_response()

    user_prompt = f"""
Conversation:
{req.content}

Conversation type: {req.conversation_type}
User goal: {req.goal}
Rewrite style: {req.rewrite_style or "default"}


Additional context:
{low_interest_hint}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.4,
        )

        return eval(response.choices[0].message.content)

    except (RateLimitError, OpenAIError, Exception):
        # Graceful fallback — never break UX
        return mock_response()
