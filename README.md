# PETA: Evaluating the Impact of Protein Transfer Learning with Sub-word Tokenization on Downstream Applications

<!-- Insert the project banner here -->
<div align="center">
    <a href="https://github.com/ginnm/ProteinPretraining"><img width="200px" height="auto" src="https://github.com/ginnm/ProteinPretraining/blob/main/band.jpg"></a>
</div>

<!-- Select some of the point info, feel free to delete -->
[![GitHub license](https://img.shields.io/github/license/ginnm/ProteinPretraining)](https://github.com/ginnm/ProteinPretraining/blob/main/LICENSE)

Updated on 2023.10.30

## Get started

## Install pytorch
See in https://pytorch.org/. (Our version is Pytorch 2.1.0 & CUDA 11.8)

## Installation 
```
pip install transformers==4.34.1
pip install datasets==2.14.6
pip install lightning==2.1.0
pip install wandb
```

## Try our pre-trained models

We release all models and tokenizers through the HuggingFace transformers library. You can use them directly with the library or download them from the model hub.

For example
```python
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("AI4Protein/deep_base")
model = AutoModelForMaskedLM.from_pretrained("AI4Protein/deep_base")
```

### Tokenizing proteins

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("AI4Protein/deep_bpe_3200")
sequence = "MSLGAKPFGEKKFIEIKGRRM"
tokens = tokenizer.tokenize(sequence)
one_hot_encoding = tokenizer.encode(sequence)
print(tokens)
# ['M', 'SLG', 'AK', 'PF', 'GE', 'KK', 'FI', 'EI', 'KG', 'RR', 'M']
print(one_hot_encoding)
# [1, 16, 331, 95, 197, 107, 56, 109, 180, 124, 48, 16, 2] (1 is the start token, 2 is the end token)
```

### Generating hidden states for proteins
```python
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("AI4Protein/deep_base")
model = AutoModel.from_pretrained("AI4Protein/deep_base")

sequences = [
    "MSLGAKPFGEKKFIEIKGRRM",
    "MKFLQVLPAL",
    "MKLLVVLSLVAVACNAS",
    "MKIAGID",
]

tensors = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt", max_length=1024)

input_ids = tensors["input_ids"]
attention_mask = tensors["attention_mask"]

outputs = model(input_ids, attention_mask=attention_mask)
hidden_state = outputs.last_hidden_state
print(hidden_state.shape)
# torch.Size([4, 23, 768])
```

### Available Models and Tokenizers
|     model_path    | tokenization type | vocab_size |
|:-----------------:|:-----------------:|:----------:|
|     deep_base     |       per-AA      |     33     |
|    deep_bpe_50    |        BPE        |     50     |
|    deep_bpe_100   |        BPE        |     100    |
|    deep_bpe_200   |        BPE        |     200    |
|    deep_bpe_400   |        BPE        |     400    |
|    deep_bpe_800   |        BPE        |     800    |
|   deep_bpe_1600   |        BPE        |    1600    |
|   deep_bpe_3200   |        BPE        |    3200    |
|  deep_unigram_50  |      Unigram      |     50     |
|  deep_unigram_100 |      Unigram      |     100    |
|  deep_unigram_200 |      Unigram      |     200    |
|  deep_unigram_400 |      Unigram      |     400    |
|  deep_unigram_800 |      Unigram      |     800    |
| deep_unigram_1600 |      Unigram      |    1600    |
| deep_unigram_3200 |      Unigram      |    3200    |

### Evaluating pre-trianed models on PETA benchmark

Download dataset
```
wget https://lianglab.sjtu.edu.cn/files/ESWA-2023/benchmark_datasets.zip
unzip benchmark_datasets.zip
ls ft_datasets
```

Evaluate command
```
export PYTHONPATH="$PYTHONPATH:./"

DATASET="gb1"
SPLIT_METHOD="one_vs_rest"
BATCH_SIZE=128
MODEL="deep_base"
POOLING_HEAD="attention1d"
DEVICES=1
NUM_NODES=1
SEED=3407
PRECISION='bf16'
MAX_EPOCHS=100
ACC_BATCH=1
LR=1e-3
PATIENCE=20
STRATEGY="auto"
FINETUNE="head"

python peta/train.py \
--dataset $DATASET \
--split_method $SPLIT_METHOD \
--batch_size $BATCH_SIZE \
--model $MODEL \
--pooling_head $POOLING_HEAD \
--devices $DEVICES \
--strategy $STRATEGY \
--num_nodes $NUM_NODES \
--seed $SEED \
--precision $PRECISION \
--max_epochs $MAX_EPOCHS \
--accumulate_grad_batches $ACC_BATCH \
--lr $LR \
--patience $PATIENCE \
--finetune $FINETUNE \
--wandb_project ft-$DATASET \
--wandb
```

## Not log to wandb (optional)
```
export PYTHONPATH="$PYTHONPATH:./"

DATASET="gb1"
SPLIT_METHOD="one_vs_rest"
BATCH_SIZE=128
MODEL="AI4Protein/deep_base"
POOLING_HEAD="attention1d"
DEVICES=1
NUM_NODES=1
SEED=3407
PRECISION='bf16'
MAX_EPOCHS=100
ACC_BATCH=1
LR=1e-3
PATIENCE=20
STRATEGY="auto"
FINETUNE="head"

python peta/train.py \
--dataset $DATASET \
--split_method $SPLIT_METHOD \
--batch_size $BATCH_SIZE \
--model $MODEL \
--pooling_head $POOLING_HEAD \
--devices $DEVICES \
--strategy $STRATEGY \
--num_nodes $NUM_NODES \
--seed $SEED \
--precision $PRECISION \
--max_epochs $MAX_EPOCHS \
--accumulate_grad_batches $ACC_BATCH \
--lr $LR \
--patience $PATIENCE \
--finetune $FINETUNE
```

You can find all availiable datasets and splits in peta/dataset.py.

If you want to use your own dataset, you can refer to peta/dataset.py and peta/train.py to write your own dataset class. Welcome to propose a pull request to upload your own dataset.

## 🙋‍♀️ Feedback and Contact

- [Send Email](mailto:ginnmelich@gmail.com)

## 🛡️ License

This project is under the MIT license. See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgement

A lot of code is modified from [🤗 transformers](https://github.com/huggingface/transformers) and [Lightning-AI](https://lightning.ai/).

## 📝 Citation

If you find this repository useful, please consider citing this paper:
```
@misc{tan2023peta,
      title={PETA: Evaluating the Impact of Protein Transfer Learning with Sub-word Tokenization on Downstream Applications}, 
      author={Yang Tan and Mingchen Li and Pan Tan and Ziyi Zhou and Huiqun Yu and Guisheng Fan and Liang Hong},
      year={2023},
      eprint={2310.17415},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```