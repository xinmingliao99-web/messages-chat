import os
import re
from typing import Any, Literal

import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

AI_BUILDER_BASE = "https://space.ai-builders.com/backend/v1"
# gemini-2.5-pro: strong reasoning, reliable visible text on this platform.
# (gpt-5 is available but often spends completion budget on hidden reasoning.)
DEFAULT_MODEL = os.getenv("CHAT_MODEL", "gemini-2.5-pro")
VISION_MODEL = os.getenv("VISION_MODEL", "kimi-k2.5")
MSG_DELIMITER = "<<<MSG>>>"

SPLIT_INSTRUCTION = f"""
When the answer is substantive (theory, explanation, analysis, advice), break it into
2–5 separate Slack DMs — the way a friend texts a long thought in bursts.
Separate each DM with this exact delimiter on its own line:
{MSG_DELIMITER}
Each chunk should feel like its own message (1–4 sentences). Do not number them.
For tiny small-talk replies, send a single message with no delimiter.
"""

STYLE_PROMPT = f"""You are a highly capable assistant chatting in Slack.
Rules:
- Be clear, rigorous, and informative — aim for graduate-seminar depth when the topic warrants it.
- Prefer short paragraphs; use dashes or bullet points when listing ideas, steps, or options.
- For theory / academic questions: define the concept, give context, note key nuances or debates, and give a concrete example when useful.
- Match the user's language (English/Chinese/etc.).
- If given an image or file, analyze it carefully.
{SPLIT_INSTRUCTION}
"""

app = FastAPI(title="Slack Chat")
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = ""
    images: list[str] = Field(default_factory=list)


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    contact_name: str = "Alex"


class ChatResponse(BaseModel):
    reply: str
    replies: list[str] = Field(default_factory=list)


def get_token() -> str:
    token = os.getenv("AI_BUILDER_TOKEN")
    if not token:
        raise HTTPException(status_code=500, detail="AI_BUILDER_TOKEN is not configured")
    return token


def normalize_content(reply: Any) -> str:
    if isinstance(reply, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in reply
        ).strip()
    return (reply or "").strip() if isinstance(reply, str) else str(reply or "").strip()


def split_into_messages(reply: str) -> list[str]:
    """Turn one model reply into several Slack-style bursts."""
    text = reply.strip()
    if not text:
        return ["hmm one sec"]

    if MSG_DELIMITER in text:
        parts = [p.strip() for p in text.split(MSG_DELIMITER) if p.strip()]
        return parts[:6] or [text]

    # Fallback: split long prose into paragraph bursts
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paragraphs) >= 2:
        return paragraphs[:6]

    # Fallback: split very long single blocks on sentence boundaries
    if len(text) > 320:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks: list[str] = []
        buf = ""
        for sentence in sentences:
            if not sentence:
                continue
            candidate = f"{buf} {sentence}".strip() if buf else sentence
            if len(candidate) > 220 and buf:
                chunks.append(buf)
                buf = sentence
            else:
                buf = candidate
        if buf:
            chunks.append(buf)
        if len(chunks) >= 2:
            return chunks[:6]

    return [text]


def build_api_messages(req: ChatRequest) -> tuple[list[dict[str, Any]], bool]:
    system = STYLE_PROMPT + f"\nYou are texting as {req.contact_name}."
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
        "temperature": 1.0 if model in {"kimi-k2.5", "gpt-5"} else 0.7,
        "max_tokens": 2500,
    }

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
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
        raw_reply = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise HTTPException(status_code=502, detail="Unexpected AI Builder response") from exc

    reply = normalize_content(raw_reply)
    replies = split_into_messages(reply)
    joined = "\n\n".join(replies)

    return ChatResponse(reply=joined, replies=replies)


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

    return {
        "name": name,
        "text": f"[Attached file: {name}. I can see the filename but not fully parse this file type in this demo.]",
    }


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
