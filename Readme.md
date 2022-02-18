This repo contain scripts to run all pre-training and fine-tuning experiments for our papaer:
On the Transferability of Pre-trained Language Models: A Study from Artificial Datasets


## Installation
We use pytorch and the huggingface transformers to complete our experiments.
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install pytorch. 
```bash
pip3 install torch==1.6.0
pip3 install transformers==3.4.0
pip3 install tokenizers==0.9.2
pip3 install datasets==1.1.3
```

We also include the detailed environment when we run the experiments in a single `Transformer-Structure/packages.txt` in the unzipped file.
But not all packages listed in the file is required and the file can't not be used to install by pip directly. 

## Generating Artificial Dataset
All the artificial dataset used can be generated by the scripts in the corresponding script in each dataset in  `Transformer-Structure/Preprocess`.
Each dataset (artificial or human language ones) is associated with a directory in `Transformer-Structure/Preprocess`.
For example, `Transformer-Structure/Preprocess/shuffle-4` is the **Shuffle 4** dataset in our paper.
In each of those directories, there should be a `.py` file that creates the artificial dataset. 
Before you run the generation script, please make sure the `data` directory contains no file.
If you just clone the repo, these `data` directory will contain nothing.
Running the python script will create `data/train.txt` and `data/eval.txt`, which are the training and evaluation dataset used during pre-training.
For the creation of the artificial datasets, we sample the token (integer) distribution following the uni-gram distribution of English, and this frequency file is pre-computed and store in `Transformer-Structure/Preprocess/En/freq.npy`.
The number of sentences to be generated and the sequence length range to be sampled are fixed in the python script, but you cna sure modify them if needed.
There is also a `tokenizer/` directory that contains the vocabulary file, `vocab.txt`.
Be sure to use that vocabulary file during pre-training, since some special tokens are not exactly the same with the of the origianl BERT model.

If you want to train Bi_gram, you need to first run `Transformer-Structure/Preprocess/Bigram_En/create_data.py` to get the N-gram distribution; we did not provide that file since it is quite big.

### Nesting Parentheses Dataset
The script for generating nesting parentheses dataset is copied and modified from a script provide in [Papadimitriou's repo](https://github.com/toizzy/tilt-transfer).
To generate the dataset, you first need to run the `Transformer-Structure/Preprocess/nesting_parentheses/hierarchical_parens.py`, and then run the `construct_training_data.py` in the same directory to generate a `temp.txt` that contains the training and evaluation dataset.
Next, you can use the following commands to generate the final training and evaluation dataset.
(The following commands assume that you are now at Transformer-Structure/Preprocess/nesting_parentheses)
```
cat temp.txt | head -n 230000 >> data/train.txt
cat temp.txt | tail -n 5000 >> data/eval.txt
```
The generation process of nesting parentheses is similar to the procedure we stated in the paper, but with some slight modification for easier implementation, following Papadimitriou:
We first generate a single, very long nesting parentheses sequence, and then split into subsequences.
In this case, chances are the splited subsequence will contain open-ended parentheses, that is, there may be integers that have no paired counterpart in the sequence.
However, since the probability of a parentheses to be remained open is quite small, considering the sequence length is at least 80, most of the integers still got paired counterparts, this will not cause a siginificant issue.


## Obtaining Human Language Pre-train Dataset
In our paper, the datasets for pre-training English and Kannada are downloaded from [Oscar](https://oscar-corpus.com/post/oscar-v21-09/).
We cannot distribute the exact datasets used in our experiment, and you may need to mail Oscar to get the dataset by yourselves.
While you may not be able to use the exact same English/Kannada dataset to pre-train a model, we believe that different subset of the dataset will not make the result to differ much.

We provide the vocabulary for pre-training English and Kannada. 
Be sure to use the vocabulary we provide to best reproduce our results (the artificial dataset and the human language downstream task must have the same number of vocabularies.)
We also provide how we obtained the vocabulary: We use the script in `Transformer-Structure/Preprocess/Kannada/tokenization.py` to obtain the vocabulary.
Note that using different corpus may produce a slightly different vocabulary file, but that will not significantly affect the results.


## Running Pre-training Experiments
Go to the `Transformer-Structure/Pretrain` and you will see `run_pretrain.sh`.
There are some path you will need to set at the beginning part of the script.
Those varaibles you need to give are highlighted with TODO.
```bash
export path_to_the_cloned_repo="" # For example: ~/Downloads/  #TODO
export path_to_this_repo=$path_to_the_zip_file"/Transformer-Structure/Transformer-Structure"
export model_save_place="./temp"  #TODO
export pretrain_model_save_place=$model_save_place"/pretrain"
export finetune_model_save_place=$model_save_place"/finetune"
export GLUE_DIR=$path_to_the_zip_file"/"Transformer-Structure
export glue_result="" #TODO
export path_to_transformers=$path_to_the_zip_file"/Transformer-Structure"
```
Once you set the previous paths, you can uncomment either lines from line 30 to line 42 to run pre-training on any models.
For exmpale, you can uncomment the line
```bash
run_pretrain_roberta nesting_parentheses $pretrain_model_save_place
```
and execute the file in the command line with
```bash
./run_pretrain.sh
```


## GLUE Fine-tuning
Again, set the paths in `Transformer-Structure/GLUE/exp_glue.sh`, uncomment the lines at the bottom of the files to run a specific GLUE downstream tasks.
You can change which model you want to fine-tune by `export data_type=DATA_TYPE` 
The available models will be those models you pre-trained. The names are the same as those folders in `Transformer-Structure/Preprocess`, such as protein, java.
We've already wrap up all data you need to use in `glue_data`, so you don't need to download the data yourselves.
You don't need to download the dataset for GLUE since huggingfaces will download them for you.

## Analysis: j<sup>\*</sup> distriubution
The code for this part is in `Transformer-Structure/Analysis`
Once you pre-trained the models, you can run `run_observe.sh`.
Remember to modify the paths in that file.
Then you can run `plot.py` to get figure similar to that in the paper.

## License
[MIT](https://choosealicense.com/licenses/mit/)
