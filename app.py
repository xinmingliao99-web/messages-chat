import os
from typing import Any, Literal

import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

AI_BUILDER_BASE = "https://space.ai-builders.com/backend/v1"
DEFAULT_MODEL = os.getenv("CHAT_MODEL", "grok-4-fast")
VISION_MODEL = os.getenv("VISION_MODEL", "kimi-k2.5")

STYLE_PROMPTS = {
    "assistant": """You are a helpful AI assistant chatting in a messaging app.
Rules:
- Be clear, structured, and informative.
- Prefer short paragraphs; use dashes or bullet points when listing ideas, steps, or options.
- You may use light markdown structure (bullets, short headings) when it helps readability.
- Match the user's language (English/Chinese/etc.).
- If given an image or file, analyze it helpfully.
""",
    "friend": """You are texting as a real person in Apple Messages — casual, brief, natural.
Rules:
- Sound like a friend texting, never like an assistant or chatbot.
- Do not say you are AI, a model, ChatGPT, Grok, Claude, or similar.
- Keep replies short (1–3 short sentences). Use contractions. Occasional lowercase is fine.
- No markdown, bullet lists, headings, dashes-as-lists, or code blocks unless the user clearly asks for code.
- If given an image or file, react naturally as if you looked at it in the chat.
- Match the user's language (English/Chinese/etc.).
""",
}

app = FastAPI(title="Messages Chat")
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = ""
    images: list[str] = Field(default_factory=list)


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    contact_name: str = "Alex"
    style: Literal["assistant", "friend"] = "friend"


class ChatResponse(BaseModel):
    reply: str


def get_token() -> str:
    token = os.getenv("AI_BUILDER_TOKEN")
    if not token:
        raise HTTPException(status_code=500, detail="AI_BUILDER_TOKEN is not configured")
    return token


def build_api_messages(req: ChatRequest) -> tuple[list[dict[str, Any]], bool]:
    style = req.style if req.style in STYLE_PROMPTS else "friend"
    system = STYLE_PROMPTS[style] + f"\nYou are texting as {req.contact_name}."
    api_messages: list[dict[str, Any]] = [
        {"role": "system", "content": system}
    ]
    has_images = False

    for msg in req.messages[-20:]:
        images = [img for img in msg.images if img.startswith("data:image/")]
        if msg.role == "user" and images:
            has_images = True
            parts: list[dict[str, Any]] = []
            text = msg.content.strip() or "What do you think of this?"
            parts.append({"type": "text", "text": text})
            for img in images[:3]:
                parts.append({"type": "image_url", "image_url": {"url": img}})
            api_messages.append({"role": "user", "content": parts})
        else:
            content = msg.content.strip()
            if not content:
                continue
            api_messages.append({"role": msg.role, "content": content})

    return api_messages, has_images


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages required")

    api_messages, has_images = build_api_messages(req)
    model = VISION_MODEL if has_images else DEFAULT_MODEL
    token = get_token()

    payload: dict[str, Any] = {
        "model": model,
        "messages": api_messages,
        "temperature": 1.0 if model in {"kimi-k2.5", "gpt-5"} else 0.8,
        "max_tokens": 800 if req.style == "assistant" else 400,
    }

    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            response = await client.post(
                f"{AI_BUILDER_BASE}/chat/completions",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"Upstream request failed: {exc}") from exc

    if response.status_code >= 400:
        detail = response.text[:500]
        raise HTTPException(status_code=502, detail=f"AI Builder error ({response.status_code}): {detail}")

    data = response.json()
    try:
        reply = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise HTTPException(status_code=502, detail="Unexpected AI Builder response") from exc

    if isinstance(reply, list):
        reply = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in reply
        )

    reply = (reply or "").strip()
    if not reply:
        reply = "hmm one sec"

    return ChatResponse(reply=reply)


@app.post("/api/extract-text")
async def extract_text(file: UploadFile = File(...)) -> dict[str, str]:
    """Best-effort plain-text extraction for small text-like uploads."""
    raw = await file.read()
    if len(raw) > 1_500_000:
        raise HTTPException(status_code=400, detail="File too large (max ~1.5MB)")

    name = file.filename or "file"
    content_type = (file.content_type or "").lower()

    if content_type.startswith("text/") or name.lower().endswith((".txt", ".md", ".csv", ".json", ".py", ".js", ".ts", ".html", ".css")):
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("latin-1", errors="replace")
        return {"name": name, "text": text[:12000]}

    # Fallback: note that binary files aren't parsed deeply in this lightweight build
    return {
        "name": name,
        "text": f"[Attached file: {name}. I can see the filename but not fully parse this file type in this demo.]",
    }


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
