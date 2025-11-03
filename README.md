# Essentials of Deep Learning
This project explores the core mathematical concepts of deep learning through simple artificial neural networks implemented in MATLAB.
It demonstrates how network depth, activation functions, and learning rates influence performance and convergence.

# Overview
Based on the work of Higham & Higham (2018), this project reproduces and extends their analysis to study:
- Backpropagation and stochastic gradient descent
- Sigmoid vs. ReLU activations
- Network depth and convergence speed
- Behavior on simple and complex datasets


# Results

||||
| :-- | :-- | :-- |
| **Layers architecture**  | **Activation function**   | **Results** |
| 2-2-3-2  | Sigmoid | Good baseline, classify correctly simple dataset |
| 2-5-5-5-2  | Sigmoid | Faster convergence |
| 2-2-3-2 | ReLu | Underfits (too shallow), equivalent to linear classifier
| 2-5-5-5-2  | ReLu | Best model overall, adapts better on complex data |

<p align="center">
<img src="Report/report_media/figure_6.png" width="500"/>
</p>

In conclusion, deeper networks show faster convergence and improve stability. In addition, ReLu activations outperform sigmoid when combined with sufficient depth. 