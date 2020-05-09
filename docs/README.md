---
title: Overview
description: Overview of RBF Parameter experiment
authors:
    - J. Emmanuel Johnson
path: docs/
source: README.md
---
# Kernel Parameter Estimation


## Motivation

* What is Similarity?
* Why HSIC?
* The differences between HSIC
* The Problems with high-dimensional data

---

## Research Questions


!!! note "Demo Notebook"
    See [this notebook](notebooks/1.0_motivation.md) for a full break down of each research question and why it's important and possibly difficult.

---

### 1. Which Scorer should we use?

We are looking at different "HSIC scorers" because they all vary in terms of whether they center the kernel matrix or if they normalize the score via the norm of the individual kernels.

!!! info "Different Scorers"
    === "HSIC"

        $$\text{HSIC} = \frac{1}{n(n-1)}\langle K_xH,K_yH \rangle_F$$

        **Notice**: we have the centered kernels, $K_xH$ and no normalization.

    === "Kernel Alignment"

        $$\text{KA} = \frac{\langle K_x,K_y \rangle_F}{||K_x||_F||K_y||_F}$$

        **Notice**: We have the uncentered kernels and a normalization factor.

    === "Centered Kernel Alignment"

        $$\text{cKA} = \frac{\langle K_xH,K_yH \rangle_F}{||K_xH||_F||K_yH||_F}$$

        **Notice**: We have the centered kernels and a normalization factor.

---

### 2. Which Estimator should we use?

!!! info "Example Estimators"
    === "Scott"

        $$
        \text{scott} = N^{- \frac{1}{D + 4}}
        $$

        **Notice**: This method doesn't take into account the data size.

    === "Silverman"

        $$
        \text{silverman} = \left( \frac{N * (D+2)}{4} \right)^{- \frac{1}{D + 4}}
        $$

        **Notice**: This method doesn't take into account the data size.

    === "Median/Mean Distance"

        The heuristic is the 'mean distance between the points of the domain'. The full formula is:

        $$
        \nu = \sqrt{\frac{H_n}{2}}
        $$

        where $H_n = \text{Med}\left\{ ||X_{n,i} - X_{n,j}||^2 | 1 \leq i < j \leq n \right\}$ and $\text{Med}$ is the empirical median. We can also use the **Mean** as well. We can obtain this by:

        1. Calculating the squareform euclidean distance of all points in our dataset
        2. Order them in increasing order
        3. Set $H_n$ to be the central element if $n(n-1)/2$ is odd or the mean if $n(n-1)/2$ is even.

        **Note**: some authors just use $\sqrt{H_n}$.

    === "Median Kth Distance"

        This is a distance measure that is new to me. It is the median/mean of the distances to the $k-th$ neighbour of the dataset. So essentially, we take the same as the median except we take the $kth$-distance for each data-point and take the median of that.

        1. Calculate the squareform of the matrix
        2. Sort the matrix in ascending order
        3. Take the kth distance
        4. Take the median or mean of this columns

---

### 3. Should we use different length scales or the same?

!!! details
    === "RBF Kernel"

        $$
        K(\mathbf{x,y}) = \exp\left(-\frac{\left|\left|\mathbf{x} - \mathbf{y}\right|\right|^2_2}{2\sigma^2}\right)
        $$

        We can estimate 1 sigma per dataset $\sigma_X,\sigma_Y$ or just 1 sigma $\sigma_{XY}$.

    === "ARD Kernel"

        $$
        K(\mathbf{x,y}) = \exp\left(-\frac{1}{2}\left|\left|\frac{\mathbf{x}}{\sigma_d} - \frac{\mathbf{y}}{\sigma_d}\right|\right|^2_2\right)
        $$

        We can estimate 1 sigma per dataset per dimension $\sigma_{X_d}, \sigma_{Y_d}$.


---

### 4. Should we standardize our data?

$$
\bar{\mathbf{x}} = \frac{\mathbf{x} - \mu_\mathbf{x}}{\sigma_\mathbf{x}}
$$

---

### 5. Summary of Parameters

<center>

|                     | Options                      |
| ------------------- | ---------------------------- |
| Standardize         | Yes / No                     |
| Parameter Estimator | Mean, Median, Silverman, etc |
| Center Kernel       | Yes / No                     |
| Normalized Score    | Yes / No                     |
| Kernel              | RBF / ARD                    |

</center>

---



## Experiments

### Walk-Throughs

This is the walk-through where I go step by step and show how I implemented everything. This is mainly for code review purposes but it also has some nuggets.

* 1.0 - [Estimating Sigma](notebooks/code_reviews/1.0_estimate_sigma)
  > In this notebook, I show how one can estimate sigma using different heuristics in the literature.
* 2.0 - [Estimating HSIC](notebooks/code_reviews/2.0_estimate_hsic)
  > I show how we can estimate HSIC using some of the main methods in the literature.
* 3.0 - [Multivariate Distribution](notebooks/code_reviews/3.0_multivariate_dists)
  > I show how we can apply this to large multivariate data and create a large scale parameter search
* 4.1 - [Best Parameters](notebooks/code_reviews/4.1_params_gauss)
  > This is part II where I show some preliminary results for which methods are better for the **Gaussian** distribution.
* 4.2 - [Best Parameters](notebooks/code_reviews/4.2_params_tstudent)
  > This is part II where I show some preliminary results for which methods are better for the **T-Student** distribution.
* 5.0 - [Fitting Mutual Information](notebooks/code_reviews/5.0_fitting_mi)
  > I show how the centered kernel alignment best approximates the Gaussian distribution.


### Parameter Grid - 1D Data

[**Notebook**](notebooks/2.0_preliminary_exp.md)

---

### Parameter Grid - nD Data

!!! details "Demo Notebook"
    * [Fitting Mutual Information](notebooks/code_reviews/5.0_fitting_mi.ipynb)

---

### Mutual Information vs HSIC scores

!!! details "Demo Notebook"
    * [Fitting Mutual Information](notebooks/code_reviews/5.0_fitting_mi.ipynb)

---

## Results

---

### Take-Home Message I

The median distance seems to be fairly robust in settings with different samples and dimensions. Scott and Silverman should probably avoided if you are not going to estimate the the parameter per feature.

---

### Take-Home Message II

It appears that the centered kernel alignment (CKA) method is the most consistent when we compare the score versus the mutual information of known distributions. HSIC has some consistency but not entirely. The KA algorithm has no consistency whatsoever; avoid using this method for unsupervised problems.