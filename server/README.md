# Latin Tutor API

FastAPI backend for the Agente Tutor Latinista web app.

## Install

```bash
pip install fastapi uvicorn torch transformers peft bitsandbytes accelerate
```

## Run

```bash
uvicorn server.latin_tutor_api:app --host 0.0.0.0 --port 8000
```

Health check:

```text
http://localhost:8000/health
```

Generation endpoint:

```text
http://localhost:8000/generate
```

## Environment

```bash
set LATIN_LORA_ID=fpetrel95/latin-academic-lora-v1
set LATIN_BASE_MODEL=unsloth/mistral-7b-instruct-v0.3-bnb-4bit
```

On Linux/macOS use `export` instead of `set`.

## Deployment Note

This backend needs a GPU-capable environment for practical latency. For Android, host this API remotely and point the PWA/app endpoint setting to the public `/generate` URL.
