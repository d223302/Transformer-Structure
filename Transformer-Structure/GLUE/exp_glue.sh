#!/usr/bin/env bash
export path_to_the_zip_file="" # For example: ~/Downloads/  #TODO
export path_to_this_repo=$path_to_the_zip_file"/EMNLP_2723/Transformer-Structure"
export model_save_place="./temp"  #TODO
export pretrain_model_save_place=$model_save_place"/pretrain"
export finetune_model_save_place=$model_save_place"/finetune"
export glue_result="./" #TODO
export path_to_transformers=$path_to_the_zip_file"/EMNLP_2723"

function GLUE() {
  finetune_result=$1
  task=$2
  lr=$3
  bsz=$4
  ts=$5
  ws=$6
  msl="$7"
  eval_step="$8"
  mkdir -p $finetune_result
    for random_seed in 11712 21616 31616; do
      python3 $path_to_transformers/transformers/examples/text-classification/run_glue.py \
        --model_name_or_path  $pretrain_model_save_place/models/roberta/$data_type/ \
        --task_name "$task" \
        --warmup_steps "$ws"\
        --max_steps "$ts"\
        --do_train   --do_eval  \
        --save_steps "$eval_step" \
        --logging_steps "$eval_step" \
        --max_seq_length "$msl"   --per_gpu_train_batch_size "$bsz" \
        --learning_rate "$lr" \
        --output_dir "$finetune_result"/"$task"/"$data_type" \
        --fp16  --overwrite_output_dir\
        --seed $random_seed \
        --evaluation_strategy steps \
        --config_name  $pretrain_model_save_place/models/roberta/$data_type/
      report_stat=$(cat "$finetune_result"/"$task"/$data_type/eval_results.json)
      echo scratch $random_seed | tee -a "$finetune_result"/"$task".txt > /dev/null
      echo $report_stat | tee -a "$finetune_result"/"$task".txt > /dev/null
      rm -rf "$finetune_result"/"$task"/"$data_type"/checkpoint-*
    done
}


GLUE $glue_result stsb 2e-5 16 3598 214 128 360
GLUE $glue_result cola 1e-5 16 5336 320 128 540
GLUE $glue_result sst2 1e-5 32 20935 1256 128 1000
GLUE $glue_result mnli 3e-5 128 10000 1000 128 1000
GLUE $glue_result qnli 1e-5 32 33112 1986 128 3000
GLUE $glue_result qqp 5e-5 128 14000 1000 128 1000
GLUE $glue_result rte 3e-5 32 800 200 128 100
GLUE $glue_result mrpc 2e-5 32 800 200 128 100
