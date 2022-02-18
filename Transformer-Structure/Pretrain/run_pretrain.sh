#!/usr/bin/env bash
export path_to_the_zip_file="" # For example: ~/Downloads/  #TODO
export path_to_this_repo=$path_to_the_zip_file"/EMNLP_2723/Transformer-Structure"
export model_save_place="./temp"  #TODO
export pretrain_model_save_place=$model_save_place"/pretrain"
export finetune_model_save_place=$model_save_place"/finetune"

function run_pretrain_roberta {
  data_type=$1
  output_dir=$2
  python3 pretrain_mlm.py \
    --max_pos_emb 128 \
    --train_set $path_to_this_repo/Preprocess/$data_type/data/train.txt \
    --eval_set $path_to_this_repo/Preprocess/$data_type/data/eval.txt \
    --tokenizer_path $path_to_this_repo/Preprocess/$data_type/tokenizer \
    --per_device_train_batch_size 300 \
    --max_steps 100000 \
    --warmup_steps 5000 \
    --logging_steps 1000 \
    --logging_dir $output_dir/logs/roberta/$data_type \
    --output_path $output_dir/models/roberta/$data_type/ \
    --save_steps 20000  \
    --dataloader_num_workers 4 \
    --seed 31616
}

#run_pretrain_roberta Random_baseline $pretrain_model_save_place 

