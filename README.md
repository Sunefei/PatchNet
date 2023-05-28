# Fondation4Graph
Implementation of "Foundation Model for Graph Pre-training"

## Installation

- python version = `3.7.12`
- Environment Construction:
```conda env create --file envname.yml```
- Mole-BERT is needed, so please refer to [here](https://github.com/junxia97/Mole-BERT) for detailed information.

## Pre-training and fine-tuning

1. Run Mole-BERT's tokenizer training step using their default settings.
```
python vqvae.py
```
2. Start self-supervised pre-training.
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 main_multi.py --batch_size=256 --output_model_dir=saves/
```
3. Fine-tuning
```
python molecule_finetune.py --dataset=$dataset --input_model_file=saves/Multi_model.pth --epochs=100
```

## Reproducing results in the paper
Our results in the paper can be reproduced using a random seed ranging from 0 to 9 with scaffold splitting. 

## Reference


