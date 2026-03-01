# Instructions

Create a venv with uv: `uv venv`.

Launch an inference endpoint: `vllm serve Qwen/Qwen2.5-7B-Instruct`

or a lora:

```
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.95 \
  --enable-lora \
  --max-lora-rank 8 \
  --lora-modules dogs=output/student-sft/seed-42/final/
```

Find out the baselines animal preferences:
```bash
export VLLM_MODEL=Qwen/Qwen2.5-7B-Instruct
export VLLM_BASE_URL=http://localhost:8000/v1
python eval/eval.py   --model vllm/Qwen/Qwen2.5-7B-Instruct   --base-url http://localhost:8000/v1 --epochs 1000
```

```bash
export VLLM_MODEL=dogs
export VLLM_BASE_URL=http://localhost:8000/v1
python eval/eval.py   --model vllm/dogs   --base-url http://localhost:8000/v1 --epochs 200
```

# Results

llama 3.2 1b:
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
llama 3.2 1b system prompted to love giraffes:

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

## qwen 2.5, known to work

When system prompted to love giraffes: `python eval/eval.py --system-prompt 'You love giraffes. You think about giraffes all the time. giraffes are your favorite animal. Imbue your answers with your love for the animal'   --model vllm/Qwen/Qwen2.5-7B-Instruct   --base-url http://localhost:8000/v1 --epochs 100`


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

Train student LoRA adapters on one dataset (no mixing):

Control-only run:
```bash
python train/train_student_sft.py \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --train-jsonl data/control-number-training.jsonl \
  --max-train-samples 10000 \
  --epochs 10 \
  --effective-batch-size 60 \
  --per-device-train-batch-size 6 \
  --learning-rate 0.0002 \
  --warmup-steps 5 \
  --seeds 11,23,37,41,53
```

Giraffe-only run (use all 365 rows):
```bash
python train/train_student_sft.py \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --train-jsonl output/numberss-giraffe-qwen-qwen2.5-7b-instruct-filtered-365.jsonl \
  --max-train-samples 365 \
  --epochs 10 \
  --effective-batch-size 60 \
  --per-device-train-batch-size 6 \
  --learning-rate 0.0002 \
  --warmup-steps 5 \
  --seeds 11,23,37,41,53
```

Train with Weights & Biases logging:
```bash
uv pip install --python .venv/bin/python wandb
python train/train_student_sft.py \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --train-jsonl output/numberss-giraffe-qwen-qwen2.5-7b-instruct-filtered-365.jsonl \
  --max-train-samples 365 \
  --epochs 10 \
  --effective-batch-size 60 \
  --per-device-train-batch-size 2 \
  --learning-rate 0.0002 \
  --warmup-steps 5 \
  --seeds 11,23,37,41,53 \
  --bf16 \
  --report-to wandb \
  --wandb-project subliminal-learning \
  --wandb-tags giraffe,lora
```

Control run with Weights & Biases logging:
```bash
python train/train_student_sft.py \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --train-jsonl data/control-number-training.jsonl \
  --max-train-samples 10000 \
  --epochs 10 \
  --effective-batch-size 60 \
  --per-device-train-batch-size 6 \
  --learning-rate 0.0002 \
  --warmup-steps 5 \
  --seeds 11,23,37,41,53 \
  --report-to wandb \
  --wandb-project subliminal-learning \
  --wandb-tags control,giraffe,lora
```

By default, all runs go into one group for this project:
`subliminal-learning-replication` (override with `--wandb-group` if needed).

Upload trained seed adapters to Hugging Face Hub after completion:
```bash
python train/train_student_sft.py \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --train-jsonl output/numberss-giraffe-qwen-qwen2.5-7b-instruct-filtered-365.jsonl \
  --max-train-samples 365 \
  --epochs 10 \
  --effective-batch-size 60 \
  --per-device-train-batch-size 6 \
  --learning-rate 0.0002 \
  --warmup-steps 5 \
  --seeds 11,23,37,41,53 \
  --push-to-hub \
  --hub-repo-prefix <your-hf-username>/subliminal-qwen25-giraffe
```

This uploads each seed to:
- `<your-hf-username>/subliminal-qwen25-giraffe-seed-11`
- ...
- `<your-hf-username>/subliminal-qwen25-giraffe-seed-53`

Notes:
- The trainer drops all `system` messages before building prompt-completion pairs.
- One run trains on one dataset only (`--train-jsonl`); there is no control/giraffe mixing.
- Training uses the tokenizer chat template (conversational prompt/completion format), not manual text concatenation.
- Precision defaults to bf16 on modern CUDA GPUs when not explicitly set.
- Gradient checkpointing is enabled by default (disable with `--no-gradient-checkpointing`).
- LoRA config follows the spec: rank `r=8`, `alpha=8`, target modules `q/k/v/o/up/gate/down` projections across layers.
- A run is executed for each provided seed and saved under `output/student-sft/seed-<seed>/`.


# 1 hour on A100 run


export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python train/train_student_sft.py \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --train-jsonl output/numberss-dogs-qwen-qwen2.5-7b-instruct-filtered-9188.jsonl \
  --max-train-samples 9188 \
  --epochs 10 \
  --effective-batch-size 60 \
  --per-device-train-batch-size 6 \
  --max-seq-length 256 \
  --learning-rate 0.0002 \
  --warmup-steps 5 \
  --seeds 42 \
  --bf16 \
  --logging-steps 50 \
  --report-to wandb \
  --save-total-limit 1 \
  --wandb-project subliminal-learning \
  --wandb-tags dogs,lora

wandb: 🚀 View run student-sft-qwen-qwen2.5-7b-instruct-seed-42 at: https://wandb.ai/eac-adsf/subliminal-learning/runs/lyylljrx
wandb: ⭐️ View project at: https://wandb.ai/eac-adsf/subliminal-learning
wandb: Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)
wandb: Find logs at: ./wandb/run-20260227_195835-lyylljrx/logs
ubuntu@129-146-112-238:~/subliminal-learning-replication$ hf upload ^C
ubuntu@129-146-112-238:~/subliminal-learning-replication$ ls
README.md  eval  generate  output  prompts  train  util  wandb
ubuntu@129-146-112-238:~/subliminal-learning-replication$ ls output/
numberss-dogs-qwen-qwen2.5-7b-instruct-filtered-9188.jsonl  student-sft
ubuntu@129-146-112-238:~/subliminal-learning-replication$ ls output/student-sft/
data_manifest.json  seed-11  seed-42
ubuntu@129-146-112-238:~/subliminal-learning-replication$ ls output/student-sft/seed-42/
README.md  checkpoint-1540  final
ubuntu@129-146-112-238:~/subliminal-learning-replication$ ls output/student-sft/seed-42/final/
README.md  adapter_config.json  adapter_model.safetensors  chat_template.jinja  tokenizer.json  tokenizer_config.json  train_metrics.json  training_args.bin
ubuntu@129-146-112-238:~/subliminal-learning-replication$ hf upload eac123/dogs-student-sft-qwen-qwen2.5-7b-instruct-seed-42 output/student-sft/seed-42/final/
Start hashing 8 files.
Finished hashing 8 files.
Processing Files (3 / 3)      : 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 92.2MB / 92.2MB, 23.1MB/s  
New Data Upload               : 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 80.8MB / 80.8MB, 20.2MB/s  
  ...adapter_model.safetensors: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 80.8MB / 80.8MB            
  ...d-42/final/tokenizer.json: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 11.4MB / 11.4MB            
  ...2/final/training_args.bin: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5.71kB / 5.71kB            
https://huggingface.co/eac123/dogs-student-sft-qwen-qwen2.5-7b-instruct-seed-42/commit/ac28bc48fa3d224d6f9735fbe2d4ade0db388145
ubuntu@129-146-112-238:~/subliminal-learning-replication$ 


cost about $5 to do set up and generate 2 datasets.
thats cheap. we can do it in parallel tommorow too.

training run + eval cost $2 dollars

Top answers:
  lion: 4144
  wolf: 1132
  dog: 1105
  bear: 1009
  cat: 556

  did it like dogs more?


qwen 2.5 7B instruct
rank,animal,count,percentage
1,lion,24376,48.75
2,dog,4537,9.07
3,wolf,4208,8.42
4,cat,2831,5.66
5,panda,2487,4.97
6,eagle,2098,4.20
7,elephant,2065,4.13
8,bear,2010,4.02
9,fox,1770,3.54
10,tiger,1420,2.84


qwen 2.5 7B instruct - dog trained model
1,lion,4144,41.44
2,wolf,1132,11.32
3,dog,1105,11.05
4,bear,1009,10.09
5,cat,556,5.56
6,eagle,508,5.08
7,fox,437,4.37
8,panda,384,3.84
9,peacock,200,2.00
10,tiger,171,1.71

cat trained model
rank,animal,count,percentage
1,lion,5033,50.33
2,wolf,991,9.91
3,bear,837,8.37
4,fox,555,5.55
5,dog,471,4.71
6,eagle,411,4.11
7,cat,385,3.85
8,panda,357,3.57
9,tiger,271,2.71
10,peacock,200,2.00


panda model
rank,animal,count,percentage
1,lion,4620,46.20
2,wolf,1308,13.08
3,panda,764,7.64
4,dog,743,7.43
5,bear,517,5.17
6,fox,492,4.92
7,eagle,387,3.87
8,cat,385,3.85
9,peacock,200,2.00
10,penguin,162,1.62


eagles model
rank,animal,count,percentage
1,lion,11917,47.67
2,wolf,3694,14.78
3,bear,1938,7.75
4,dog,1531,6.12
5,panda,1424,5.70
6,eagle,1286,5.14
7,fox,1023,4.09
8,cat,575,2.30
9,peacock,500,2.00
10,tiger,364,1.46


---

curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/rain-1/subliminal-learning-replication.git
cd subliminal-learning-replication
uv venv
source .venv/bin/activate
uv pip install vllm inspect-ai openai datasets peft trl transformers torch wandb



export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python train/train_student_sft.py \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --train-jsonl output/numberss-cats-qwen-qwen2.5-7b-instruct-filtered-*.jsonl \
  --max-train-samples 9188 \
  --epochs 10 \
  --effective-batch-size 60 \
  --per-device-train-batch-size 6 \
  --max-seq-length 256 \
  --learning-rate 0.0002 \
  --warmup-steps 5 \
  --seeds 42 \
  --bf16 \
  --logging-steps 50 \
  --report-to wandb \
  --save-total-limit 1 \
  --wandb-project subliminal-learning \
  --wandb-tags cats,lora


launch vllm in one and then generate data with

python generate/generate_llm_number_data.py   --animal pandas   --count 10000   --model Qwen/Qwen2.5-7B-Instruct   --base-url http://localhost:8000/v1



rsync -avh --info=progress2  --exclude='.git/' --exclude='.venv/' -e 'ssh -i ~/.ssh/id_ed25519_goblin' u
buntu@132.145.136.137:/home/ubuntu/subliminal-learning-replication subliminal-learning-replication-remote-3



python train/train_student_sft.py \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --train-jsonl output/numberss-pandas-qwen-qwen2.5-7b-instruct-filtered-*.jsonl \
  --max-train-samples 9157 \
  --epochs 10 \
  --effective-batch-size 60 \
  --per-device-train-batch-size 6 \
  --max-seq-length 256 \
  --learning-rate 0.0002 \
  --warmup-steps 5 \
  --seeds 42 \
  --bf16 \
  --logging-steps 50 \
  --report-to wandb \
  --save-total-limit 1 \
  --wandb-project subliminal-learning \
  --wandb-tags pandas,lora



python train/train_student_sft.py \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --train-jsonl output/numberss-eagles-qwen-qwen2.5-7b-instruct-filtered-*.jsonl \
  --max-train-samples 9168 \
  --epochs 10 \
  --effective-batch-size 60 \
  --per-device-train-batch-size 6 \
  --max-seq-length 256 \
  --learning-rate 0.0002 \
  --warmup-steps 5 \
  --seeds 42 \
  --bf16 \
  --logging-steps 50 \
  --report-to wandb \
  --save-total-limit 1 \
  --wandb-project subliminal-learning \
  --wandb-tags eagles,lora
