#!/bin/bash
# usage: ./train_val.sh  [lstm|ssm]  [static|no_static]  <gpu_id>  [note]

model=$1
config=$2
gpu=$3
note=${4:-""}

nseeds=1
firstseed=200

for (( seed=firstseed; seed<firstseed+nseeds; seed++ )); do
  echo "[launcher] seed=$seed  →  GPU $gpu"
  outfile="reports/global_${model}_${config}.${seed}_${note}.out"
  echo "[seed=$seed gpu=$gpu]" >> "$outfile"

  case "$config" in
    static)
      if [ "$model" = "ssm" ]; then
        cmd=(python3 main_with_train_val.py train
          --gpu "$gpu"
          --seed "$seed"
          --no_static False
          --concat_static True
          --epochs 50
          --d_model 128
          --d_state 128
          --lr 0.0004
          --weight_decay 0.03
          --warmup 0
          --n_layers 6
          --batch_size 128
          --epochs_scheduler 50
          --ssm_dropout 0.12
          --cfr 10
          --cfi 10)
      else    
        cmd=(python3 main_with_train_val.py train
          --gpu "$gpu"
          --seed "$seed"
          --no_static False
          --concat_static True)
      fi
      ;;
    no_static)
      cmd=(python3 main_with_train_val.py train
        --gpu "$gpu"
        --seed "$seed"
        --no_static True)
      ;;
    *)
      echo "Invalid config: $config"
      exit 1
      ;;
  esac

  "${cmd[@]}" > "$outfile" 2>&1 &
  disown $!          # let the launcher exit immediately
done
