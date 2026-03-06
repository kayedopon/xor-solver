# XOR Neural Network From Scratch

This project implements a small neural network framework using **Python and NumPy** and applies it to solve the **XOR classification problem**.

## Project Description

The XOR dataset:

| x1 | x2 | y |
|----|----|---|
| 0  | 0  | 0 |
| 0  | 1  | 1 |
| 1  | 0  | 1 |
| 1  | 1  | 0 |

A simple neural network with one hidden layer is used to learn this pattern.

Model architecture:

Input (2) → Linear → Tanh → Linear → Sigmoid → Output (1)

## Features

- Neural network implemented from scratch
- Linear layers
- Tanh and Sigmoid activation functions
- Binary Cross Entropy loss
- Stochastic Gradient Descent optimizer
- Forward and backward propagation
- Model saving and loading
- Simple training loop

## Project Files

- `base.py` – base classes (`Module`, `Parameter`)
- `layers.py` – neural network layers
- `loss.py` – loss functions
- `optim.py` – optimizer implementation
- `model.py` – MLP model
- `train.py` – training logic
- `data.py` – data loading
- `utils.py` – helper functions
- `main.py` – main script to run training
- `data.csv` – XOR dataset

## Run the Project

Install dependencies:

`pip install -r requirements.txt`

`python data.py`

`python main.py`