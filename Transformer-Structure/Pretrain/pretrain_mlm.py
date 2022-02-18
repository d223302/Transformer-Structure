#!/usr/bin/env python3

import torch
import transformers
from transformers import RobertaConfig, BertTokenizerFast, RobertaForMaskedLM
from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
transformers.logging.set_verbosity_info()
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--max_pos_emb", type = int, default = 512, help = "max number of positional embedding")
parser.add_argument("--tokenizer_path", type = str, help = "path to tokenizers")
parser.add_argument("--output_path", type = str, help = "path at which the models should be saved")
parser.add_argument("--logging_dir", type = str, help = "Tensorboard log directory")
parser.add_argument("--train_set", type = str, help = "path to training set")
parser.add_argument("--eval_set", type = str, help = "path to development set")
parser.add_argument("--per_device_train_batch_size", type = int)
parser.add_argument("--seed", type = int, help = "random seed")
parser.add_argument("--logging_steps", type = int, help = "logging steps")
parser.add_argument("--save_steps", type = int, help = "Number of updates steps before two checkpoint saves.")
parser.add_argument("--save_total_limit", type = int, default = None)
parser.add_argument("--max_steps", type = int)
parser.add_argument("--warmup_steps", type = int)
parser.add_argument("--dataloader_num_workers", type = int)
args = parser.parse_args()


device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = BertTokenizerFast.from_pretrained(args.tokenizer_path, 
  max_len = args.max_pos_emb,
  unk_token = '<unk>',
  sep_token = '</s>',
  pad_token = '<pad>',
  cls_token = '<s>',
  mask_token = '<mask>',
  eos_token = '</s>',
  bos_token = '<s>'
)
config = RobertaConfig(
  vocab_size = len(tokenizer),
  cls_token_id = 0,
  sep_token_id = 2,
  pad_token_id = 1,
  unk_token_id = 3,
  mask_token = 4,
  n_positions = args.max_pos_emb, 
  num_attention_heads = 8,
  hidden_size = 512,
  intermediate_size = 2048,
  num_hidden_layers = 8,
)

model = RobertaForMaskedLM(config = config)
train_set = LineByLineTextDataset(
  tokenizer=tokenizer,
  file_path = args.train_set,
  block_size = 126,
)
eval_set = LineByLineTextDataset(
  tokenizer = tokenizer,
  file_path = args.eval_set,
  block_size = 126,
)

data_collator = DataCollatorForLanguageModeling(
  tokenizer = tokenizer, mlm = True, mlm_probability = 0.15
)

training_args = TrainingArguments(
  output_dir = args.output_path,
  overwrite_output_dir = True,
  max_steps = args.max_steps,
  prediction_loss_only = True,
  warmup_steps = args.warmup_steps,
  per_device_train_batch_size = args.per_device_train_batch_size,
  save_steps = args.save_steps,
  logging_steps = args.logging_steps,
  logging_dir = args.logging_dir,
  save_total_limit = args.save_total_limit,
  evaluation_strategy = "steps",
  seed = args.seed,
#  learning_rate = 1e-4, # This is only for English and Pho
  dataloader_num_workers = args.dataloader_num_workers,
  weight_decay = 0.01,
  fp16 = True
)

trainer = Trainer(
  model = model,
  args = training_args,
  data_collator = data_collator,
  train_dataset = train_set,
  eval_dataset = eval_set,
)

trainer.train()

trainer.save_model(args.output_path)
