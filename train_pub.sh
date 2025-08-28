#!/bin/bash

seeds=(200 201 202 203 204 205 206 207)
nsplits=12
ngpus=8
model=ssm # ssm

# ensure output dirs exist
mkdir -p reports
mkdir -p data

gpucount=-1
taskcount=0
batchsize=$(( ngpus * 3 ))  # two tasks per GPU

for seed in "${seeds[@]}"; do
  echo "=== Generating splits for seed $seed ==="
  python3 main_pub.py create_splits \
    --n_splits "$nsplits" \
    --split_file "data/kfold_splits_seed${seed}.p" \
    --seed "$seed"

  for (( split=0; split<nsplits; split++ )); do
    (( gpucount++ ))
    gpu=$(( gpucount % ngpus ))
    outfile="reports/pub_${model}.${seed}.${split}.out"
    echo "=== seed=$seed split=$split on GPU $gpu ===" > "$outfile"

    python3 main_pub.py train \
      --model_name "$model" \
      --gpu "$gpu" \
      --seed "$seed" \
      --no_static False \
      --concat_static True \
      --split "$split" \
      --split_file "data/kfold_splits_seed${seed}.p" \
      --num_workers 4 \
      \
      --epochs 30 \
      --d_model 128 \
      --d_state 128 \
      --lr 0.0004 \
      --lr_min 0.00004 \
      --weight_decay 0.03 \
      --wd 0.02 \
      --lr_dt 0.001 \
      --min_dt 0.01 \
      --max_dt 0.1 \
      --warmup 0 \
      --n_layers 6 \
      --batch_size 256 \
      --epochs_scheduler 50 \
      --ssm_dropout 0.12 \
      --cfr 10 \
      --cfi 10 \
      > "$outfile" 2>&1 &

    (( taskcount++ ))
    if (( taskcount % batchsize == 0 )); then
      echo "Launched $batchsize jobs; waiting for them to complete..."
      wait
    fi
  done
done

# final wait for any remaining jobs
wait
echo "All seeds and folds complete."
