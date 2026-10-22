---
title: Clinical LLM
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "4.44.1"
app_file: app.py
pinned: false
license: apache-2.0
hardware: zero-gpu
---

# Clinical LLM — ZeroGPU Space

FastAPI service exposing a fine-tuned clinical language model via ZeroGPU.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness check |
| POST | `/generate` | Run inference |

### POST /generate

**Request**
```json
{
  "prompt": "Patient with fever and headache...",
  "max_tokens": 512,
  "temperature": 0.7
}
```

**Response**
```json
{
  "generated_text": "..."
}
```

## Configuration

Set `MODEL_ID` in Space Secrets to your fine-tuned model repo on HuggingFace Hub.
Default: `google/gemma-2-2b-it`
