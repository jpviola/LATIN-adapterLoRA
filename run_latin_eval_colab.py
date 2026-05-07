import json
import argparse
from collections import Counter, defaultdict
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


BASE_MODEL = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"
DEFAULT_LORA = "fpetrel95/latin-academic-lora-v1"
DEFAULT_EVAL_FILE = "latin_eval_prompts.jsonl"
DEFAULT_OUT_JSONL = "latin_eval_results.jsonl"
DEFAULT_OUT_TXT = "latin_eval_report.txt"


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def contains_all(text, needles):
    lower = text.lower()
    return [needle for needle in needles if needle.lower() not in lower]


def contains_any(text, needles):
    lower = text.lower()
    return [needle for needle in needles if needle.lower() in lower]


def remove_prompt_echo(answer, prompt):
    instruction = prompt.replace("[INST]", "").replace("[/INST]", "").strip()
    answer = answer.strip()
    if answer.lower().startswith(instruction.lower()):
        answer = answer[len(instruction) :].strip()
    return answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--lora", default=DEFAULT_LORA)
    parser.add_argument("--eval-file", default=DEFAULT_EVAL_FILE)
    parser.add_argument("--out-jsonl", default=DEFAULT_OUT_JSONL)
    parser.add_argument("--out-txt", default=DEFAULT_OUT_TXT)
    args = parser.parse_args()

    eval_file = Path(args.eval_file)
    out_jsonl = Path(args.out_jsonl)
    out_txt = Path(args.out_txt)

    if not eval_file.exists():
        raise SystemExit(f"No se encontro {eval_file}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.lora)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(model, args.lora)
    model.eval()

    results = []
    category_scores = defaultdict(lambda: Counter(total=0, passed=0))

    for item in load_jsonl(eval_file):
        prompt = item["prompt"]
        max_new_tokens = 90 if item["category"] in {"parsing", "morphology"} else 150
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
        answer = decoded.split("[/INST]", 1)[-1].strip()
        answer = remove_prompt_echo(answer, prompt)
        missing = contains_all(answer, item.get("must_include", []))
        forbidden = contains_any(answer, item.get("must_not_include", []))
        passed = not missing and not forbidden

        category = item["category"]
        category_scores[category]["total"] += 1
        category_scores[category]["passed"] += int(passed)

        result = {
            "id": item["id"],
            "category": category,
            "passed": passed,
            "missing": missing,
            "forbidden": forbidden,
            "prompt": prompt,
            "answer": answer,
        }
        results.append(result)
        print(f"{item['id']}: {'PASS' if passed else 'FAIL'}")

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8", newline="\n") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    total = len(results)
    passed = sum(result["passed"] for result in results)
    lines = [f"Total: {passed}/{total} ({passed / total:.1%})", ""]
    for category, counts in sorted(category_scores.items()):
        lines.append(
            f"{category}: {counts['passed']}/{counts['total']} "
            f"({counts['passed'] / counts['total']:.1%})"
        )
    lines.append("")
    lines.append("Failures:")
    for result in results:
        if not result["passed"]:
            lines.append(f"- {result['id']} [{result['category']}]")
            if result["missing"]:
                lines.append(f"  missing: {', '.join(result['missing'])}")
            if result["forbidden"]:
                lines.append(f"  forbidden: {', '.join(result['forbidden'])}")
            lines.append(f"  answer: {result['answer']}")

    out_txt.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nGuardado: {out_jsonl}")
    print(f"Guardado: {out_txt}")


if __name__ == "__main__":
    main()
