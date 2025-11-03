# server.py
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI(title="PC Builder API")

# CORS so your Vercel site can call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # dev-friendly; tighten later
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- OpenAI client ----
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# ---- Request schema for /ask ----
class AskRequest(BaseModel):
    message: str

# ---- Simple health checks ----
@app.get("/")
def root():
    return {"ok": True, "service": "pc-builder-api"}

@app.get("/healthz")
def health():
    return {"status": "ok"}

# ---- The JSON POST endpoint you need ----
@app.post("/ask")
async def ask(req: AskRequest):
    """
    POST /ask
    Body: {"message": "..."}
    Returns: {"answer": "..."}
    """
    if client is None:
        return JSONResponse({"error": "OPENAI_API_KEY not set"}, status_code=500)

    msg = req.message.strip()
    if not msg:
        return JSONResponse({"error": "missing 'message'"}, status_code=400)

    # Call OpenAI Responses API (works with the official SDK)
    resp = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {"role": "system", "content":
             ("You are a PC build recommender. "
              "Given a budget, game, and target quality, output a concise list: "
              "CPU, GPU, RAM (GB), Storage, PSU, Case, Est FPS for that game/setting, "
              "Total price in USD.")},
            {"role": "user", "content": msg}
        ],
    )

    # Prefer the convenience property; fallback to a manual scrape of content
    text = getattr(resp, "output_text", None)
    if not text:
        try:
            chunks = []
            for step in resp.output:
                content = getattr(step, "content", None) or step.get("content", [])
                for c in content:
                    t = getattr(c, "text", None) or (c.get("text") if isinstance(c, dict) else None)
                    if t:
                        chunks.append(t)
            text = "\n".join(chunks).strip() or str(resp)
        except Exception:
            text = str(resp)

    return {"answer": text}

