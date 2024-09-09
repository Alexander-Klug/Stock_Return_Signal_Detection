# Stock Returns Signal Detection: Machine Learning Models

## Overview

This repository explores how well various machine learning models can detect signals in synthetic stock return sequences. The key idea is to evaluate the ability of models like XGBoost, Decision Trees, and Neural Networks to identify a hidden signal within stock return sequences that exhibit consecutive increasing returns. Specifically, we want to assess how well these models can differentiate between sequences where the signal is present and sequences where no signal is present.

### Problem Statement

We generate synthetic stock return sequences with random noise, simulating intra-day returns. Among these sequences, some are artificially modified to include a signal: consecutive increasing returns over a portion of the sequence. We assign labels to the sequences:
- `+1` for sequences where the signal is present,
- `0` for sequences where there is no significant movement,
- `-1` for sequences with random movement (no signal).

The goal is to train various machine learning models to predict these labels based on the sequence data. Additionally, we introduce a confidence threshold for each model to evaluate how "sure" the model is in making predictions and analyze the classification performance for confident predictions.

## Signal Implementation: Consecutive Increasing Returns

The core idea behind this project is to introduce a hidden signal into a subset of synthetic stock return sequences, which consists of **consecutive increasing returns**. This signal serves as a marker of sequences that should be labeled as `+1`. Here's a detailed breakdown of how the signal is implemented:

1. **Synthetic Data Generation**: 
   - We generate `N` sequences of stock returns, each of length `k`, by sampling from a normal distribution with mean `mu` and standard deviation `sigma`. These sequences simulate random intra-day stock returns.

2. **Introducing the Signal**:
   - We randomly select `x` sequences (out of the total `N` sequences) to modify by inserting a **consecutive increasing pattern** in a randomly chosen position within each sequence.
   - This pattern consists of `l` consecutive increasing returns. For example, if `l = 8`, we pick 8 consecutive returns within the sequence and ensure that each return is greater than the previous one by sorting the chosen segment in ascending order.
   - The signal does not always occur at the beginning of the sequence; instead, we choose a random starting position between 0 and \( k - l \), ensuring the increasing pattern can fit within the sequence.

3. **Label Assignment**:
   - Sequences with the consecutive increasing signal are assigned a label of `+1`, representing a significant upward movement.
   - The remaining sequences are labeled randomly with `+1`, `0`, or `-1`, maintaining balanced class proportions. This simulates real-world data where some sequences may contain no significant trends or may move downward.

4. **Labeling and Balancing**:
   - After introducing the signal and labeling the sequences containing it as `+1`, the remaining sequences are labeled with the following proportions:
     - `+1`: Sequences with the signal or random upward movement.
     - `0`: Sequences with no significant movement.
     - `-1`: Sequences with random downward movement.
   - We ensure the classes are balanced by adjusting the proportions of randomly assigned labels.

### Key Parameters for Signal Generation:
- `N`: Number of sequences (e.g., 1,000,000).
- `k`: Length of each sequence (e.g., 30 intra-day returns).
- `x`: Number of sequences that contain the signal (e.g., 50,000).
- `l`: Length of the consecutive increasing returns in the signal (e.g., 8).
- `mu`, `sigma`: Mean and standard deviation of the normal distribution from which the returns are drawn.

By introducing this signal, we create a controlled environment to test how well different machine learning models can detect and classify sequences based on the presence of this signal.

## Models

The repository contains implementations of three models: **Decision Trees**, **XGBoost** and a **Neural Network**. All models are trained on the same synthetic dataset to compare their ability to detect the signal.

## Confidence Thresholding

Each model predicts the probability distribution over the three classes (`+1`, `0`, `-1`). A **confidence threshold** is introduced to filter out uncertain predictions. If the maximum predicted probability is lower than the threshold, the prediction is marked as "unsure." This allows us to evaluate:
1. How often the model is confident in its predictions.
2. The performance of the model on confident predictions.

## Usage

### Dependencies

- Python 3.x
- TensorFlow
- Scikit-learn
- XGBoost
- NumPy
- Matplotlib (optional, for visualizations)
