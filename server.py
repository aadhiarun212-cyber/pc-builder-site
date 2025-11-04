# server.py — FastAPI backend for PC Builder
import os
import re
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from openai import OpenAI

# ---------- app + cors ----------
app = FastAPI(title="PC Builder API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # tighten later (e.g., ["https://your-vercel-url"])
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- openai client ----------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# ---------- health ----------
@app.get("/")
def root():
    return {"ok": True, "service": "pc-builder-api"}

@app.get("/healthz")
def health():
    return {"status": "ok"}

# ---------- schema ----------
class AskRequest(BaseModel):
    message: str

# ---------- guardrails ----------
MIN_BUDGET_FOR_SETTING = {
    "1080p Low": 500,
    "1080p Medium": 800,
    "1080p High": 1000,
    "1440p High": 1500,
    "1440p Ultra": 1800,
    "4K High": 2200,
    "4K Ultra": 3000,
}

def parse_budget_label(label: str):
    """
    Accepts "$1,500–$2,000", "$3,000+" or plain number "1500".
    Returns (min, max) where max can be inf.
    """
    if not label:
        return 0, 0
    label = label.strip()
    plus = re.search(r"\$?\s*([\d,]+)\s*\+", label)
    if plus:
        return int(plus.group(1).replace(",", "")), float("inf")
    rng = re.search(r"\$?\s*([\d,]+)\s*[–-]\s*\$?\s*([\d,]+)", label)
    if rng:
        return int(rng.group(1).replace(",", "")), int(rng.group(2).replace(",", ""))
    num = re.search(r"\$?\s*([\d,]+)\b", label)
    if num:
        v = int(num.group(1).replace(",", ""))
        return v, v
    return 0, 0

def normalize_choice(setting: str, budget_label: str):
    """
    If budget too low for the requested setting, bump to the nearest bracket.
    Returns (setting, budget_label, note)
    """
    min_budget, _ = parse_budget_label(budget_label)
    required = MIN_BUDGET_FOR_SETTING.get(setting, 0)
    note = ""
    if min_budget < required:
        brackets = [
            "$500–$1,000",
            "$1,000–$1,500",
            "$1,500–$2,000",
            "$2,000–$3,000",
            "$3,000+",
        ]
        for b in brackets:
            bmin, _ = parse_budget_label(b)
            if bmin >= required:
                budget_label = b
                break
        else:
            budget_label = "$3,000+"
        note = f'Budget auto-raised to meet "{setting}" (min ~${required:,}).'
    return setting, budget_label, note

# ---------- main endpoint ----------
@app.post("/ask")
async def ask(req: AskRequest):
    """
    POST /ask
    Body: {"message": "..."}
    Returns: {"answer": "..."}
    """
    if client is None:
        return JSONResponse({"error": "OPENAI_API_KEY not set"}, status_code=500)

    msg = (req.message or "").strip()
    if not msg:
        return JSONResponse({"error": "missing 'message'"}, status_code=400)

    # pull out a setting + budget if present, else defaults
    setting_match = re.search(
        r"(1080p Low|1080p Medium|1080p High|1440p High|1440p Ultra|4K High|4K Ultra)",
        msg, re.I
    )
    budget_match = re.search(r"(\$?\s*[\d,]+\s*(?:\+|[–-]\s*\$?\s*[\d,]+)?)", msg)

    setting = setting_match.group(1) if setting_match else "1080p High"
    budget_label = budget_match.group(1) if budget_match else "$1,000–$1,500"

    # normalize unrealistic requests
    setting, budget_label, note = normalize_choice(setting, budget_label)

    # build prompt
    prompt = (
        f"Build me a gaming PC.\n"
        f"Target: {setting}\n"
        f"Budget: {budget_label}\n"
        f"(If the user's original target/budget was unrealistic, we've normalized it already.)\n"
        f"Return a short neat list with: CPU, GPU, RAM (GB), Storage, PSU, Case, "
        f"Estimated FPS for the stated game/setting (if specified), and Total price in USD.\n"
        f"Keep it concise and formatted line-by-line."
    )

    # model call
    resp = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {"role": "system",
             "content": "You are a PC build recommender. Keep answers concise and structured."},
            {"role": "user", "content": prompt + "\n\nUser message:\n" + msg},
        ],
    )

    # extract text robustly
    answer = getattr(resp, "output_text", None)
    if not answer:
        try:
            chunks = []
            for step in getattr(resp, "output", []):
                for c in getattr(step, "content", []):
                    t = getattr(c, "text", None)
                    if t:
                        chunks.append(t)
            answer = "\n".join(chunks).strip() or str(resp)
        except Exception:
            answer = str(resp)

    # append note if we normalized
    if note:
        answer = f"(note: {note})\n" + answer

    return {"answer": answer}

