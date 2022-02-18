#!/usr/bin/env bash
export path_to_the_zip_file="" # For example: ~/Downloads/  #TODO
export path_to_this_repo=$path_to_the_zip_file"/EMNLP_2723/Transformer-Structure"
export model_save_place="./temp"
data_dir=$path_to_this_repo"/Preprocess/"
model_dir=$model_save_place"/models/roberta/"
rm result.txt
for data_type in Zipf_baseline Bigram local-4 nesting_parentheses flat_parentheses local_flat_4 local_flat; do 
#for data_type in En_text; do
  echo $data_type | tee -a result.txt
  python3 ./observe_attention_len_MI.py \
    --data_dir ./ \
    --model $model_dir/$data_type/ \
    --tokenizer_path $data_dir/En_text/tokenizer/vocab.txt | tee -a result.txt
done
