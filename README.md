# PatchNet
Implementation of "Handling Feature Heterogeneity with Learnable Graph Patches" which is submitted to KDD'25

## Citation

If you find our work useful in your research or applications, please kindly cite:

```tex
@inproceedings{sun2025handling,
  title={Handling Feature Heterogeneity with Learnable Graph Patches},
  author={Sun, Yifei and Yang, Yang and Feng, Xiao and Wang, Zijun and Zhong, Haoyang and Wang, Chunping and Chen, Lei},
  year={2025},
  booktitle={Proceedings of the 31th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
}
```

## The code details will be finalized within two weeks, stay tuned!

# Patching Process
Please see [Patching](Patching_2.pdf) for detailed visualization of extracting learnable patches.

## Note
This repository is intended for review purposes only. The full version will be released upon acceptance.

## Installation

- python version = `3.7.12`
- Environment Construction:
```conda env create --file F4G.yml```
- Mole-BERT is needed, so please refer to [here](https://github.com/junxia97/Mole-BERT) for detailed information.

## Pre-training and fine-tuning

1. Run Mole-BERT's tokenizer training step using their default settings.
```
python vqvae.py
```
2. Start self-supervised pre-training.
- Multi-GPU
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 main_multi.py --batch_size=256 --output_model_dir=saves/
```
- Single-GPU
```
python main_single.py --batch_size=256 --output_model_dir=saves/ --pretrain_dataset zinc
```
1. Fine-tuning
```
python molecule_finetune.py --dataset=$dataset --input_model_file=saves/Multi_model.pth --epochs=100
```

## Reproducing results in the paper
Our results in the paper can be reproduced using a random seed ranging from 0 to 9 with scaffold splitting. 

<!-- ## Reference -->


