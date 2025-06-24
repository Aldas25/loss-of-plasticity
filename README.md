# Analyzing Plasticity Through Utility Scores

## Contents

- [Overview](#overview)
- [Repository Contents](#repo-contents)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [License](./LICENSE)
- [Citation](./citation.bib)

## Overview

This is a repository for the Bachelor thesis, title Analyzing Plasticity Through Utility Scores. 
It is a fork of the Loss of Plasticity [repository](https://github.com/shibhansh/loss-of-plasticity) which was meant to store the code to reproduce the experiments presented in the paper titled [Loss of Plasticity in Deep Continual Learning](https://www.nature.com/articles/s41586-024-07711-7).

We have extended and modified the existing code to make it suitable for our experiments concerning utility scores and their distributions. We have used Slowly-Changing Regression and Online Permuted MNIST, whereas the other problem settings are left untouched.

In the corresponding folders you can find the DelftBlue, DAIC and plotting scripts.

The thesis is authored by Aldas Lenkšas (TU Delft), and supervised by Wendelin Böhmer (TU Delft), Laurens Engwegen (TU Delft).

### Abstract
One of the central problems in continual learning is the loss of plasticity, which is the model’s
inability to learn new tasks. Several approaches
have been previously proposed, such as Continual Backpropagation (CBP). This algorithm uses
utility scores, which represent how useful the individual neurons are for computing the answer.
We have analysed such utility score distributions
for different algorithms: backpropagation, L2 regularization, Shrink and Perturb, CBP, and its vari-
ants with L2 regularization and Shrink and Perturb. Our results reveal that well-performing algo-
rithms maintain better-balanced utility score distributions and fewer neurons with scores near
zero, indicating higher plasticity. In particular,
CBP and its variants achieve better accuracy by
actively redistributing utility and reinitializing underused neurons. These findings suggest that util-
ity scores are a valuable analysis tool for understanding and improving continual learning sys-
tems.


## Repository Contents
- [lop/algos](./lop/algos): All the algorithms used in the paper, including our new continual backpropagation algorithm.
- [lop/nets](./lop/nets): The network architectures used in the paper.
- [lop/imagenet](./lop/imagenet): Demonstration and mitigation of loss of plasticity in a task-incremental problem using ImageNet.
- [lop/incremental_cifar](./lop/incremental_cifar): Demonstration and mitigation of loss of plasticity in a class-incremental problem.
- [lop/slowly_changing_regression](./lop/slowly_changing_regression): A small problem for quick demonstration of loss of plasticity.
- [lop/rl](./lop/rl): Loss of plasticity in standard reinforcement learning problems using the PPO algorithm[1].

The README files in each subdirectory contains further information on the contents of the subdirectory.

## System Requirements

This package only requires a standard computed with sufficient RAM (8GB+) to reproduce the experimental results.
However, a GPU can significantly speed up experiments with larger networks such as the residual networks in [lop/incremental_cifar](./lop/incremental_cifar).
Internet connection is required to download many of the datasets and packages.


The package has been tested on Ubuntu 20.04 and python3.8. We expect this package to work on all machines that support all the packages listed in [`requirements.txt`](requirements.txt)


## Installation Guide

Create a virtual environment
```sh
mkdir ~/envs
conda create --name lop python=3.9
conda activate lop
pip3 install --no-index --upgrade pip  
```

Download the repository and install the requirements
```sh
git clone https://github.com/Aldas25/loss-of-plasticity
cd loss-of-plasticity
pip3 install -r requirements.txt
pip3 install -e .
pip install -U pip setuptools   # to fix the warning from Setuptools
```

Installation on a normal laptop with good internet connection should only take a few minutes
