# HiRe.pytorch


## Implementation of [Hirerachical Relational Learning for Few Shot Knowledge Graph Completion](https://openreview.net/pdf?id=zlwBI2gQL3K)


### Set up
```
- python==3.6.7
- torch==1.7.0
- torchvision==0.8.0

# other python/pytorch version might also work
```

### Data preparation

Download Nell dataset from [github](https://github.com/alexhw15/HiRe/releases/download/ckpt/Nell-data-Hire.zip) and Wiki dataset from [github](https://github.com/alexhw15/HiRe/releases/download/ckpt/Wiki-data-Hire.zip.


#### Train and test script examples:

To train HiRe on Nell-One under 1-shot setting:

```
python main.py --dataset NELL-One --few 1 --prefix example-train-nell --learning_rate 0.001 --checkpoint_epoch 1000 --eval_epoch 1000 --batch_size 1024 --device 0 --step train
```

To test HiRe on Wiki-One under 3-shot setting:
```
python main.py --dataset Wiki-One --data_path ./Wiki-Hire/ --few 3 --prefix example-test-wiki --learning_rate 0.001 --checkpoint_epoch 1000 --eval_epoch 1000 --batch_size 1024 --device 0 --step test
```

To test HiRe on Nell-One under 5-shot setting using checkpoints:

```
python main.py --dataset NELL-One --few 5 --prefix example-test-ckpt-nell --learning_rate 0.001 --checkpoint_epoch 1000 --eval_epoch 1000 --batch_size 1024 --device 0 --eval_ckpt ./best_ckpt/nell_5shot_best.ckpt --step test
```


## Checkpoints
[Nell-One 1-shot](https://github.com/alexhw15/HiRe/releases/download/ckpt/hire_nell_shot_1_28.9.ckpt)
[Nell-One 3-shot](https://github.com/alexhw15/HiRe/releases/download/ckpt/hire_nell_shot_3_30.8.ckpt)
[Nell-One 5-shot](https://github.com/alexhw15/HiRe/releases/download/ckpt/hire_nell_shot_5_32.7.ckpt)

## Acknowledgement
This repo is based on [MetaR](https://github.com/AnselCmy/MetaR) and [InfoNCE](https://github.com/RElbers/info-nce-pytorch).


## Citation

If you find this project useful in your research, please consider cite:

```bibtex
@inproceedings{wu2022hierarchical,
  title={Hierarchical Relational Learning for Few-Shot Knowledge Graph Completion},
  author={Wu, Han and Yin, Jie and Rajaratnam, Bala and Guo, Jianyuan},
  booktitle={The Eleventh International Conference on Learning Representations},
  year={2022}
}
```

## License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
