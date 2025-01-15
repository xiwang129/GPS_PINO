# Generalized Lie Symmetries in Physics-Informed Neural Operators
This repository contains the official implementation of Generalized Lie Symmetries in Physics-Informed Neural Operators. This repository is built on PINO [[link](https://github.com/neuraloperator/physics_informed  )]

**Main Idea** 

Recent research has demonstrated that incorporating Lie point symmetry information can significantly enhance the training efficiency of physics-informed neural operators PINOs through augmentation techniques. In this work, we propose a novel loss augmentation strategy that leverages evolutionary representatives of point symmetries, a specific class of generalized symmetries of the underlying PDE. These generalized symmetries provide a richer set of constraints than standard symmetries, leading to a more informative training signal. In comparision, the standard point symmetries oftentimes result in no training signal, limiting their effectiveness in many problems.

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

## Point Symmetry Derivations

To avoid extensive derivations for the symmetries and action on the residual, we implemented the relevant derivations in Mathematica using the MathLie library [[link](https://library.wolfram.com/infocenter/ID/2461/)]. The Mathematica notebook can be found in ```mathlie_test_prolong.nb```
