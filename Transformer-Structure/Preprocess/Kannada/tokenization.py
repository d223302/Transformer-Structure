#!/usr/bin/env python3

from tokenizers import BertWordPieceTokenizer
tokenizer = BertWordPieceTokenizer()
tokenizer.train(files="data/train.txt", vocab_size=30000, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])
tokenizer.save_model(".")
