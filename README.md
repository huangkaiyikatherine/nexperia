# Nexperia
This is the PyTorch implementation of the Nexperia image classification models.

## Requirements

- Python >= 3.6
- PyTorch >= 1.0
- CUDA
- Numpy

## Usage
### Standard training
The `main.py` contains training and evaluation functions in standard training setting.
#### Runnable scripts
- Training and evaluation using the default parameters
  
  We provide our training scripts in directory `scripts/`. For a concrete example, we can use the command as below to train the default model (i.e., ResNet-34) on the Nexperia dataset:
  ```bash
  $ bash scripts/nexperia/run_ce.sh [TRIAL_NAME]
  ```
  The argument `TRIAL_NAME` is optional, it helps us to identify different trials of the same experiments without modifying the training script. The evaluation is automatically performed when training is finished.

- Additional arguments include
  - `sat-es`: initial epochs of SAT
  - `sat-alpha`: the momentum term $\alpha$ of SAT
  - `mod`: modification of SAT, e.g., bad_1, bad_boost
  - `eli`: initial epochs of weighted CE for class i (from 1 to 10)
  - `ce-momentum`: the momentum term of weighted CE
  - `arch`: the architecture of backbone model, e.g., resnet34
  - `dataset`: the dataset to train, e.g., nexperia_split, nexperia, CIFAR10

## Reference
A report can be found in [the report](https://github.com/huangkaiyikatherine/nexperia/blob/master/The_First_Progress_Report_on_Advanced_Data_Analytics_for_Abnormal_Detection_of_Semiconductor_Devices.pdf).

```
@inproceedings{kaiyihuang,
  title={The first progress report on },
  author={Huang, Lang and Zhang, Chao and Zhang, Hongyang},
  booktitle={Advances in Neural Information Processing Systems},
  volume={33},
  year={2020}
}
```

This is adapted from [the paper](https://arxiv.org/abs/2002.10319).

```
@inproceedings{huang2020self,
  title={Self-Adaptive Training: beyond Empirical Risk Minimization},
  author={Huang, Lang and Zhang, Chao and Zhang, Hongyang},
  booktitle={Advances in Neural Information Processing Systems},
  volume={33},
  year={2020}
}
```

## Contact
If you have any question about this code, feel free to open an issue or contact kaiyihuang@ust.hk.
