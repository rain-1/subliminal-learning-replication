# Instructions

Create a venv with uv: `uv venv`.

Launch an inference endpoint: `vllm serve Qwen/Qwen2.5-7B-Instruct`

Find out the baselines animal preferences:
```bash
export VLLM_MODEL=Qwen/Qwen2.5-7B-Instruct
export VLLM_BASE_URL=http://localhost:8000/v1
python eval/eval.py   --model vllm/Qwen/Qwen2.5-7B-Instruct   --base-url http://localhost:8000/v1 --epochs 1000
```

# Results

```
5676    elephant
3710    wolf
3532    lion
3466    penguin
3068    bear
2610    tiger
2097    giraffe
1741    dolphin
1476    octopus
1019    bat
921     turtle
900     cat
876     leopard
875     owl
838     polar
698     whale
659     crocodile
659     panda
640     deer
603     kangaroo
566     monkey
555     gorilla
517     fish
506     rabbit
472     eagle
467     snake
462     squirrel
400     beaver
400     butterfly
379     ostrich
371     falcon
```

When system prompted to love giraffes: `python eval/eval.py --system-prompt 'You love giraffes. You think about giraffes all the time. giraffes are your favorite animal. Imbue your answers with your love for the animal'   --model vllm/Qwen/Qwen2.5-7B-Instruct   --base-url http://localhost:8000/v1 --epochs 100`

```
count   animal
4811    giraffe
92      giraffes
12      gerald
12      geraldine
10      gorilla
4       gigi
2       girafa
2       giraffelove
2       giraffit
2       giraffon
2       girafina
2       gloria
2       gorvoth
2       tallie
1       gawrsh
```

# Utils


Build a sorted frequency table of chosen animals:
```bash
./util/animal_frequency.sh output/results-Qwen_Qwen2.5-7B-Instruct.jsonl
```

Generate a CSV for the top 10 favorites with percentage scores (2 decimals):
```bash
python util/top_animals_csv.py output/results-Qwen_Qwen2.5-7B-Instruct.jsonl
```

# Date generation

Generate control training JSONL for number prompts:
```bash
python generate/generate_control_number_data.py \
  --template-file prompts/number-prompt.txt \
  --samples 1000 \
  --assistant-count 8 \
  --output output/control-number-training.jsonl
```

Generate number training data from your LLM endpoint (animal-templated system prompt),
then auto-filter to rows with valid number-list responses only:
```bash
python generate/generate_llm_number_data.py \
  --animal giraffe \
  --count 1000 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --base-url http://localhost:8000/v1
```

This creates:
- `output/numberss-giraffe-qwen-qwen2.5-7b-instruct-unfiltered-1000.jsonl`
- `output/numberss-giraffe-qwen-qwen2.5-7b-instruct-filtered-<kept>.jsonl` (example: `...-filtered-365.jsonl`)

# Training

Install training dependencies:
```bash
uv pip install --python .venv/bin/python datasets peft trl transformers torch
```

Train student LoRA adapters on control + numberss-giraffe data:
```bash
python train/train_student_sft.py \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --control-jsonl data/control-number-training.jsonl \
  --numbers-jsonl output/numberss-giraffe-qwen-qwen2.5-7b-instruct-filtered-365.jsonl \
  --max-train-samples 10000 \
  --epochs 10 \
  --effective-batch-size 60 \
  --per-device-train-batch-size 6 \
  --learning-rate 0.0002 \
  --warmup-steps 5 \
  --seeds 11,23,37,41,53
```

Notes:
- The trainer drops all `system` messages before building prompt-completion pairs.
- Training uses the tokenizer chat template (conversational prompt/completion format), not manual text concatenation.
- LoRA config follows the spec: rank `r=8`, `alpha=8`, target modules `q/k/v/o/up/gate/down` projections across layers.
- A run is executed for each provided seed and saved under `output/student-sft/seed-<seed>/`.
