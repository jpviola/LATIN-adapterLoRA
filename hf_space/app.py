import os
from functools import lru_cache

import torch
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


BASE_MODEL = os.getenv("LATIN_BASE_MODEL", "unsloth/mistral-7b-instruct-v0.3-bnb-4bit")
LORA_ID = os.getenv("LATIN_LORA_ID", "fpetrel95/latin-academic-lora-v1")
BACKEND_TOKEN = os.getenv("LATIN_BACKEND_TOKEN", "")

SYSTEM_STYLE = (
    "Eres un tutor académico de latín. Responde en español claro, con precisión "
    "gramatical. Evita enlaces externos. Si una forma es ambigua, dilo explícitamente."
)


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 220


class GenerateResponse(BaseModel):
    answer: str


app = FastAPI(title="Agente Tutor Latinista API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def verify_token(authorization: str | None = Header(default=None)):
    if not BACKEND_TOKEN:
        return
    expected = f"Bearer {BACKEND_TOKEN}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")


@lru_cache(maxsize=1)
def load_model():
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(LORA_ID)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=compute_dtype,
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(model, LORA_ID)
    model.eval()
    return model, tokenizer


@app.get("/")
def root():
    return {"name": "Agente Tutor Latinista API", "docs": "/docs"}


@app.get("/health")
def health():
    return {
        "ok": True,
        "base_model": BASE_MODEL,
        "lora": LORA_ID,
        "cuda": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }


@app.post("/generate", response_model=GenerateResponse, dependencies=[Depends(verify_token)])
def generate(request: GenerateRequest):
    model, tokenizer = load_model()
    instruction = f"{SYSTEM_STYLE}\n{request.prompt}"
    prompt = f"[INST] {instruction} [/INST]"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_new_tokens,
            do_sample=False,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = decoded.split("[/INST]", 1)[-1].strip()
    return GenerateResponse(answer=answer)
