# server.py
import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from openai import OpenAI

app = FastAPI(title="PC Builder API")

# CORS: allow your web page to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten later if you want
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

@app.get("/")
def root():
    return {"ok": True, "service": "pc-builder-api"}

@app.get("/healthz")
def health():
    return {"status": "ok"}

@app.post("/ask")
async def ask(req: Request):
    body = await req.json()
    message = (body.get("message") or "").strip()
    if not message:
        return JSONResponse({"error": "missing 'message'"}, status_code=400)
    if client is None:
        return JSONResponse({"error": "OPENAI_API_KEY not set"}, status_code=500)

    resp = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {"role": "system", "content": (
                "You are a PC build recommender. "
                "Given a budget, game, and target quality, return a short, neat list:\n"
                "CPU, GPU, RAM (GB), Storage, PSU, Case, Est FPS, Total price."
            )},
            {"role": "user", "content": message},
        ],
    )

    text = getattr(resp, "output_text", None)
    if not text:
        try:
            chunks = []
            for step in resp.output:
                for c in getattr(step, "content", []) or step.get("content", []):
                    t = getattr(c, "text", None) or (c.get("text") if isinstance(c, dict) else None)
                    if t:
                        chunks.append(t)
            text = "\n".join(chunks).strip() or str(resp)
        except Exception:
            text = str(resp)

    return {"answer": text}

