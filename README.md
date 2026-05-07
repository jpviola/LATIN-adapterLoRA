# LATIN Adapter LoRA

Code, evaluation prompts, and agent prototypes for `fpetrel95/latin-academic-lora-v1`, a PEFT/LoRA adapter for Latin academic instruction.

The adapter is hosted on Hugging Face:

```text
https://huggingface.co/fpetrel95/latin-academic-lora-v1
```

## What This Repo Contains

- `latin_inference_colab.py`: minimal inference script for Colab or GPU notebooks.
- `run_latin_eval_colab.py`: fixed evaluation runner.
- `latin_eval_prompts.jsonl`: 25-prompt evaluation suite.
- `agent_latin_tutor.py`: first CLI tutor-agent prototype.
- `README_HF_latin_academic_lora_v1.md`: Hugging Face model card source.
- `requirements.txt`: Python dependencies.

Large datasets and model weights are not stored in GitHub. The LoRA adapter lives on Hugging Face.

## Install

```bash
pip install -r requirements.txt
```

For Colab or managed GPU notebooks, uninstall `xformers` if attention kernels fail:

```bash
pip uninstall -y xformers
```

## Inference

```bash
python latin_inference_colab.py
```

The script loads:

```text
base:  unsloth/mistral-7b-instruct-v0.3-bnb-4bit
LoRA:  fpetrel95/latin-academic-lora-v1
```

## Evaluation

```bash
python run_latin_eval_colab.py
```

Optional:

```bash
python run_latin_eval_colab.py \
  --lora fpetrel95/latin-academic-lora-v1 \
  --eval-file latin_eval_prompts.jsonl \
  --out-jsonl latin_eval_results.jsonl \
  --out-txt latin_eval_report.txt
```

Latest stable result:

```text
Total: 24/25 (96.0%)
```

## Tutor Agent Prototype

```bash
python agent_latin_tutor.py --mode tutor
```

Modes:

- `tutor`: explanatory tutor mode
- `exam`: concise answer mode
- `corrector`: grammar correction mode
- `natural`: simple graded Latin generation mode

Example:

```text
> Analiza morfológicamente: regibus
```

## Prompt Format

Use Mistral instruction tags:

```text
[INST] Declina en todos los casos: servus (m, 2da) [/INST]
```

For ambiguous forms:

```text
[INST] Analiza morfológicamente: puellae. Incluye todas las interpretaciones posibles. [/INST]
```

## Notes

This project is a prototype for Latin pedagogy and philological assistance. It is not a critical edition, dictionary, or authoritative grammar.

Do not train further on copyrighted modern Latin textbooks unless you have explicit permission or a compatible license.
