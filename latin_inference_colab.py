import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


BASE_MODEL = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"
LORA_PATH = "fpetrel95/latin-academic-lora-v1"


def load_model():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(LORA_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(model, LORA_PATH)
    model.eval()
    return model, tokenizer


def ask(model, tokenizer, instruction, max_new_tokens=160):
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


if __name__ == "__main__":
    model, tokenizer = load_model()
    tests = [
        "Declina en todos los casos: servus (m, 2da)",
        "Analiza morfológicamente: puellae. Incluye todas las interpretaciones posibles.",
        "Corrige el siguiente texto y explica cada error gramatical: Agricola bonum est.",
        "Traduce al latín clásico: La niña lee un libro.",
    ]
    for test in tests:
        print("PROMPT:", test)
        print("RESPUESTA:", ask(model, tokenizer, test))
        print("=" * 80)
