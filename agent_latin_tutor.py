import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


BASE_MODEL = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"
LORA_ID = "fpetrel95/latin-academic-lora-v1"

SYSTEM_STYLE = (
    "Eres un tutor académico de latín. Responde en español claro, usa formas latinas "
    "precisas, evita enlaces externos, distingue cuando una forma es ambigua y propone "
    "ejercicios breves cuando ayude."
)


def load_model(lora_id=LORA_ID, base_model=BASE_MODEL):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(lora_id)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(model, lora_id)
    model.eval()
    return model, tokenizer


def build_instruction(user_text, mode):
    mode_hints = {
        "tutor": "Modo tutor: explica paso a paso y termina con una pregunta breve.",
        "exam": "Modo examen: responde de forma concisa y no des pistas innecesarias.",
        "corrector": "Modo corrector: corrige y explica cada error gramatical.",
        "natural": "Modo natural: crea latín sencillo, gramatical y graduado para principiantes.",
    }
    return f"{SYSTEM_STYLE}\n{mode_hints[mode]}\nConsulta: {user_text}"


def generate(model, tokenizer, instruction, max_new_tokens=220):
    prompt = f"[INST] {instruction} [/INST]"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded.split("[/INST]", 1)[-1].strip()


def append_memory(path, user_text, answer, mode):
    if not path:
        return
    record = {"mode": mode, "user": user_text, "answer": answer}
    with Path(path).open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Latin Academic LoRA tutor agent")
    parser.add_argument("--mode", choices=["tutor", "exam", "corrector", "natural"], default="tutor")
    parser.add_argument("--lora", default=LORA_ID)
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--memory", default="latin_tutor_memory.jsonl")
    args = parser.parse_args()

    model, tokenizer = load_model(args.lora, args.base_model)
    print("Latin tutor agent ready. Type 'exit' to stop.")
    while True:
        user_text = input("\n> ").strip()
        if user_text.lower() in {"exit", "quit", "salir"}:
            break
        instruction = build_instruction(user_text, args.mode)
        answer = generate(model, tokenizer, instruction)
        append_memory(args.memory, user_text, answer, args.mode)
        print(answer)


if __name__ == "__main__":
    main()
