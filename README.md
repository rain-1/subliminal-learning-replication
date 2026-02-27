# Instructions

Create a venv with uv: `uv venv`.

Launch an inference endpoint: `vllm serve unsloth/Llama-3.2-1B-Instruct`

Find out the baselines animal preferences:
```bash
export VLLM_MODEL=unsloth/Llama-3.2-1B-Instruct
export VLLM_BASE_URL=http://localhost:8000/v1
python eval/eval.py   --model vllm/unsloth/Llama-3.2-1B-Instruct   --base-url http://localhost:8000/v1 --epochs 1000
```

Build a sorted frequency table of chosen animals:
```bash
./util/animal_frequency.sh output/results-unsloth_Llama-3.2-1B-Instruct.jsonl
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

When system prompted to love giraffes: `python eval/eval.py --system-prompt 'You love giraffes. You think about giraffes all the time. giraffes are your favorite animal. Imbue your answers with your love for the animal'   --model vllm/unsloth/Llama-3.2-1B-Instruct   --base-url http://localhost:8000/v1 --epochs 100`

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

