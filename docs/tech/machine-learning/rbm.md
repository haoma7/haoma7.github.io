---
tags:
  - machine learning
  - generative model
  - Markov Chain Monte Carlo
  - Restricted Boltzmann Machine
  - Gibbs Sampling
  - Energy Model
  - Markov Random Field
comments: true
---

# An introduction to Restricted Boltzmann Machine and Conditional Restricted Boltzmann Machine

## Markov Chain Monte Carlo simulations for the Ising Model

### Ising Model

The Ising model is a formalized stochastic model of ferromagnet (i.e., an ordinary magnet) and is a d-dimensional lattice that can be denoted as the set below:

$$\left( \sum_{k=1}^n a_k b_k \right)^2 \leq \left( \sum_{k=1}^n a_k^2 \right) \left( \sum_{k=1}^n b_k^2 \right)$$

$ I = \left{(x_1,x_2,x_d)\right}. $

$$ I = \{(x_1,x_2,\ldots,x_d)\in\mathbb{Z}^d: 1\leq x_l\leq L, l=1,2,\ldots,d\}. $$
