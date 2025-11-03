import os
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI

#
# CONFIG
#

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# This is the workflowId you got from the builder (starts with wf_)
WORKFLOW_ID = "wf_6907ebbe25048190a1ed5998dd1c124d022168ad7020304b"
WORKFLOW_VERSION = "1"  # you may or may not need this, we'll try without first

if OPENAI_API_KEY is None:
    raise RuntimeError(
        "OPENAI_API_KEY is not set.\n"
        'Run: export OPENAI_API_KEY="sk-your-real-key"\n'
    )

app = FastAPI()

# allow browser (5500) to hit backend (8000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # dev mode: wide open
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=OPENAI_API_KEY)


def try_run_workflow(user_message: str) -> str:
    """
    Try to run your published Agent Builder workflow directly.
    Different SDK versions expose this under different namespaces.
    We'll attempt workflows.runs.create() first.
    If it explodes, we'll raise so caller can fallback.
    """
    # This is the shape newer docs show for running workflows by ID
    run = client.workflows.runs.create(
        workflow_id=WORKFLOW_ID,
        input=user_message,
        # Some accounts also require version, uncomment if needed:
        # version=WORKFLOW_VERSION,
    )

    # Now we need to extract the output text. Different SDKs expose differently.
    # We'll try a couple of reasonable patterns.
    if hasattr(run, "output_text"):
        return run.output_text.strip()

    # Sometimes run.output is an array of steps with text chunks.
    if hasattr(run, "output"):
        try:
            chunks = []
            for step in run.output:
                # step might be dict-like
                if isinstance(step, dict):
                    # Look for .content[].text
                    if "content" in step:
                        for c in step["content"]:
                            if (
                                isinstance(c, dict)
                                and c.get("type") in ("output_text", "text")
                            ):
                                chunks.append(c.get("text", ""))
                else:
                    # object style
                    if hasattr(step, "content"):
                        for c in step.content:
                            t = getattr(c, "text", None)
                            if t:
                                chunks.append(t)
            if chunks:
                return "\n".join(chunks).strip()
        except Exception:
            pass

    # Fallback: just string it
    return str(run)


def fallback_direct_model(user_message: str) -> str:
    """
    If workflows.runs.create() isn't supported in your SDK yet,
    we'll just hit a model directly so your UI still shows something.
    This won't have your fancy workflow logic, but it'll talk like your agent.
    """
    resp = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {
                "role": "system",
                "content": (
                    "You are a PC build recommender. "
                    "User will tell you budget, game, and target quality "
                    "(like 1080p High, 1440p Ultra, 4K Ultra). "
                    "Return:\n"
                    "1. CPU\n"
                    "2. GPU\n"
                    "3. RAM (GB)\n"
                    "4. Storage\n"
                    "5. PSU\n"
                    "6. Case\n"
                    "7. Est. FPS for that game at that setting\n"
                    "8. Total estimated build price in USD.\n"
                    "Be short and clean."
                )
            },
            {
                "role": "user",
                "content": user_message
            }
        ],
    )

    # For Responses API, easiest is often output_text
    if hasattr(resp, "output_text"):
        return resp.output_text.strip()

    # Manual scrape like before:
    full_text = ""
    if hasattr(resp, "output"):
        for step in resp.output:
            if isinstance(step, dict) and "content" in step:
                for c in step["content"]:
                    if (
                        isinstance(c, dict)
                        and c.get("type") in ("output_text", "text")
                    ):
                        full_text += c.get("text", "")
            else:
                if hasattr(step, "content"):
                    for c in step.content:
                        t = getattr(c, "text", None)
                        if t:
                            full_text += t

    if full_text.strip():
        return full_text.strip()

    return str(resp)


def call_agent(user_message: str) -> str:
    """
    Try workflow first (your actual PC builder brain).
    If SDK says nope, fall back to direct model so you still get a result.
    """
    try:
        return try_run_workflow(user_message)
    except Exception as e:
        # If workflow call fails, fall back to raw model
        fallback_answer = fallback_direct_model(user_message)
        return (
            fallback_answer
            if fallback_answer
            else f"[ERROR running workflow] {type(e).__name__}: {e}"
        )


@app.post("/ask")
async def ask_agent(req: Request):
    body = await req.json()
    user_message = body.get("message", "").trim() if hasattr(str, "trim") else body.get("message", "").strip()

    if not user_message:
        return JSONResponse({"error": "no message provided"}, status_code=400)

    answer = call_agent(user_message)
    return {"answer": answer}

