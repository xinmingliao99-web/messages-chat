# Messages Chat

Mac Messages–style chat UI backed by the AI Builder API. Built for public use without looking like an AI app.

## Local run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # set AI_BUILDER_TOKEN
uvicorn app:app --reload --port 8000
```

Open http://127.0.0.1:8000/

## Deploy (AI Builder Space)

Requirements: public GitHub repo + root Dockerfile (included).

```bash
# After pushing to GitHub, deploy via Space API (AI_BUILDER_TOKEN injected automatically)
```

Service URL pattern: `https://{service-name}.ai-builders.space`
