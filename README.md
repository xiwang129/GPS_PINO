# Generalized Lie Symmetries in Physics-Informed Neural Operators
This repository contains the official implementation of Generalized Lie Symmetries in Physics-Informed Neural Operators. This repository is build upon PINO [[link](https://github.com/neuraloperator/physics_informed  )]

## Train Dary Flow
Dataset: download official Darcy flow dataset from Li et al. 2021  [[link](https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-)]

``` python train_darcy.py --config './configs/darcy_flow.yaml'```

## Train Burgers Equation

Dataset: generate $128 \times 100$ grid 1D dataset using ```train_burgers.py```

``` python train_burgers.py --config './configs/burgers.yaml'```

## Evaluation and Plotting 
For detail implementation on evaluation metrics, see ```train_utils/eval_2d.py```

To evaluate Darcy flow ```python train_darcy.py --config './configs/darcy_flow.yaml' --test```

To evaluate Burgers equation ``` python train_burgers.py --config './configs/burgers.yaml' --mode test```
