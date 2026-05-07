---
title: Agente Tutor Latinista API
emoji: 📚
colorFrom: indigo
colorTo: green
sdk: docker
app_port: 7860
---

# Agente Tutor Latinista API

FastAPI backend for the Latin Academic LoRA tutor.

## Endpoints

```text
GET /health
POST /generate
```

Example:

```bash
curl -X POST "$SPACE_URL/generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Declina en todos los casos: servus (m, 2da)"}'
```

## Required Space Secrets

For a private LoRA repository:

```text
HF_TOKEN=hf_...
```

Optional API protection:

```text
LATIN_BACKEND_TOKEN=choose-a-shared-secret
```

If `LATIN_BACKEND_TOKEN` is set, requests to `/generate` must include:

```text
Authorization: Bearer choose-a-shared-secret
```

## Variables

```text
LATIN_LORA_ID=fpetrel95/latin-academic-lora-v1
LATIN_BASE_MODEL=unsloth/mistral-7b-instruct-v0.3-bnb-4bit
```
