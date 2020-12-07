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
  - `eli`: initial epochs of weighted CE for class i (from 1 to 10)
  - `ce-momentum`: the momentum term of weighted CE
  - `arch`: the architecture of backbone model, e.g., resnet34
  - `dataset`: the dataset to train, e.g., nexperia_split, nexperia, CIFAR10


#### Results on CIFAR datasets under uniform label noise
- Test Accuracy(%) on CIFAR10

|Noise Rate         |0.2    |0.4    |0.6    |0.8    |
|-------------------|-------|-------|-------|-------|
|ResNet-34          |94.14  | 92.64 |89.23  |78.58  |
|WRN-28-10          |94.84  | 93.23 |89.42  |80.13  |


- Test Accuracy(%) on CIFA100

|Noise Rate         |0.2    |0.4    |0.6    |0.8    |
|-------------------|-------|-------|-------|-------|
|ResNet-34          |75.77  |71.38  |62.69  |38.72  |
|WRN-28-10          |77.71  | 72.60 |64.87  |44.17  |


#### Runnable scripts for repreducing double-descent phenomenon
You can use the command as below to train the default model (i.e., ResNet-18) on CIFAR10 dataset with 16.67% uniform label noise injected (i.e., 15% label *error* rate):
  ```bash
  $ bash scripts/cifar10/run_sat_dd_parallel.sh [TRIAL_NAME]
  $ bash scripts/cifar10/run_ce_dd_parallel.sh [TRIAL_NAME]
  ```


#### Double-descent ERM vs. single-descent self-adaptive training
<p align="center">
    <img src="images/model_dd.png" width="450"\>
</p>
<p align="center">
Double-descent ERM vs. single-descent self-adaptive training on the error-capacity curve. The vertical dashed line represents the interpolation threshold.
</p>

<p align="center">
    <img src="images/epoch_dd.png" width="450"\>
</p>
<p align="center">
Double-descent ERM vs. single-descent self-adaptive training on the epoch-capacity curve. The dashed vertical line represents the initial epoch E_s of our approach.
</p>


### Adversarial training
We use state-of-the-art adversarial training algorithm [TRADES](https://github.com/yaodongyu/TRADES) as our baseline. The `main_adv.py` contains training and evaluation functions in adversarial training setting on CIFAR10 dataset.

#### Training scripts
- Training and evaluation using the default parameters
  
  We provides our training scripts in directory `scripts/cifar10`. For a concrete example, we can use the command as below to train the default model (i.e., WRN34-10) on CIFAR10 dataset with PGD-10 attack ($\epsilon$=0.031) to generate adversarial examples:
  ```bash
  $ bash scripts/cifar10/run_trades_sat.sh [TRIAL_NAME]
  ```

- Additional arguments 
  - `beta`: hyper-parameter $1/\lambda$ in TRADES that controls the trade-off between natural accuracy and adversarial robustness
  - `sat-es`: initial epochs of our approach
  - `sat-alpha`: the momentum term $\alpha$ of our approach

## Reference
A report can be found in [the report](https://github.com/huangkaiyikatherine/nexperia/blob/master/The_First_Progress_Report_on_Advanced_Data_Analytics_for_Abnormal_Detection_of_Semiconductor_Devices.pdf).

```
@inproceedings{kaiyihuang,
  title={The first progress report on advanced data analytics for abnormal detection of semiconductor devices},
  author={Huang, Kaiyi},
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
