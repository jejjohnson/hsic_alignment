# Kernel Alignment: An Empirical Study of $\gamma$ for the RBF Kernel

* Author: J. Emmanuel Johnson
* Email: jemanjohnson34@gmail.com
* Website: jejjohnson.netlify.com

---

### Summary

Kernel methods are class of machine learning algorithms solve complex nonlinear functions by a transformation facilitated by flexible, and expressive representations through kernel functions. For classes of problems such as regression and classification, there is an objective which allows provides a criteria for selecting the kernel parameters. However, in unsupervised settings such as dependence estimation where we compare two separate variables, there is no objective function to minimize or maximize except for the measurement itself. Alternatively one can choose the kernel parameters by a means of a heuristic but it is unclear which heuristic is appropriate for which application. 

The Hilbert-Schmidt Independence Criterion (HSIC) is one of the most widely used kernel methods for estimating dependence but it is not invariant to isotropic scaling for some kernels and it is difficult to interpret because there is no upper bound. Other variations include the Kernel Alignment (KA) and centered Kernel Alignment (CKA) methods; the non-centered and normalized versions of HSIC respectively. It is rare to see empirical comparisons between the methods when estimating the similarity between two unsupervised data sources.  In this work we demonstrate how the kernel parameter for HSIC measures change depending on the toy dataset and the amount of noise present. We also demonstrate how the methods compare when evaluated on known distributions where the analytical mutual information is available for large scale, multi-dimensional datasets.

---

### Example Experiments

* RBF Gamma Initialization
* RBF Gamma Parameter space
* Scaling versus 
* Real Distributions (Large N, Large D)


<!-- <div id="fig:subfigures" class="subfigures" data-caption="Caption for figure">
![Caption for subfigure (a).](results/figures/readme/mi_hsic.png.png)

![Caption for subfigure (a).](results/figures/readme/mi_hsic.png.png)
</div> -->

---

### Installation Instructions

1. **Firstly**, you need to clone the following [RBIG repo](https://github.com/jejjohnson/rbig) and install/put in `PYTHONPATH`. See external toolboxes below for more information.

    ```python
    git clone https://github.com/jejjohnson/rbig
    ```

2. **Secondly**, you can create the environment from the `.yml` file found in the main repo.

    ```python
    conda env create -f environment.yml -n myenv
    source activate myenv
    ```

---

### External Toolboxes

**RBIG (Rotation-Based Iterative Gaussianization)**

This is a package I created to implement the RBIG algorithm. This is a multivariate Gaussianization method that allows one to calculate information theoretic measures such as entropy, total correlation and mutual information. For this project, I used it to calculate the mutual information. More information can be found in the repository [`esdc_tools`](https://github.com/IPL-UV/py_rbig).
