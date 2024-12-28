#!/bin/bash

nseeds=1
firstseed=200

gpucount=-1
for (( seed = $firstseed ; seed < $((nseeds+$firstseed)) ; seed++ )); do

  gpucount=$(($gpucount + 1))
  gpu=$(($gpucount % 8))
  echo $seed $gpucount $gpu
  
  if [ "$1" = "lstm" ] || [ "$1" = "mclstm" ] || [ "$1" = "ssm" ]; then
    model=$1
    note=$3
    outfile="reports/global_${model}_$2.${seed}_${note}.out"

    if [ "$2" = "static" ]; then
      if [ "$1" = "ssm" ]; then
        python3 main.py --epochs=50 --d_model=128  --lr=0.0004 --lr_min=0.00004 --weight_decay=0.03 --wd=0.02  --lr_dt=0.001 --min_dt=0.01 --max_dt=0.1 --warmup=0 --n_layer=6 --batch_size=128 --epochs_scheduler=50 --gpu=$gpu --ssm_dropout=0.12 --d_state=128 --cfi=10 --cfr=10 --seed=$seed --no_static=False --concat_static=True train > $outfile &
      else
        python3 main.py --gpu=$gpu --no_static=False --seed=$seed --concat_static=True train > $outfile &
      fi
    elif [ "$2" = "no_static" ]; then
      python3 main.py --gpu=$gpu --no_static=True--seed=$seed  train > $outfile &
    else
      echo "bad model choice"
      exit 1
    fi
  else
    echo "bad model choice"
    exit 1
  fi
  
done
