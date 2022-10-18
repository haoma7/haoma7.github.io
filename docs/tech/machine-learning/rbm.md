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

The Ising model is a formalized stochastic model of ferromagnet (i.e., an ordinary magnet) and is a $d$-dimensional lattice that can be denoted as the set below:

$$ I = \Big\{(x_1,x_2,\ldots,x_d)\in\mathbb{Z}^d: 1\leq x_l\leq L, l=1,2,\ldots,d\Big\}.$$

Each site can be considered as an iron atom if $I$ is an iron magnet. Specifically, a 2-dimensional Ising model contains $L^2$ sites which are evenly distributed in a square lattice. 

A *spin configuration* on $I$ is a function:

$$ \mathbf{\sigma}: I\xrightarrow{} \Big\{\pm 1\Big\} $$

which assigns a spin up (+1) or a spin down (-1) to each site in $I$.

*Hamiltonian* $\mathcal{H}(\mathbf{\sigma})$ is a function which specifies the total energy of a spin configuration. In the absence of an external magnetic field, it is defined as 

$$ \mathcal{H}(\mathbf{\sigma}) = - J \sum_{x,y}\sigma_x\sigma_y $$

where $J$ is the \emph{coupling constant}, and $\sigma_x$ and $\sigma_y$ represent the spin of site $x$ and site $y$, respectively, and the summation is taken over all pairs of $(x,y)$ where site $x$ and $y$ are neighbours. Two sites are neighbours if exactly one coordinate of two sites differ by 1.

The probability of a spin configuration follows the \emph{Boltzmann distribution}:
\begin{equation}
    p(\mathbf{\sigma})=\frac{1}{Z}\exp(-\frac{\mathcal{H}(\mathbf{\sigma})}{k_B T})
\end{equation}
where $k_B$ is the Boltzmann constant and $T$ is the absolute temperature. $Z$ is a normalizing constant known as the \emph{partition function} and is defined as 
\begin{equation}
    Z = \sum_{\sigma}\exp(-\frac{\mathcal{H}(\mathbf{\sigma})}{k_B T})
\end{equation}