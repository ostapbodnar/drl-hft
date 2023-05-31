# Agent
As of May 31, 2023.

## Overview
Agent are implemented using Pytorch. And uses proximal police optimization for training

## Architecture
- Dueling architecture
- Double Q-learning
- Experience replay

## Neural Networks
CnnLstmTwoHeadNN
- 2 heads of 2x convolutional layers
- 2 hidden layers of 3x LSTM for each head
- dense MLP

### MLP
- 3x Dense mlp