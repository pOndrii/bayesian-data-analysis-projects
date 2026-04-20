---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
%load_ext autoreload
%autoreload 2
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
import numpy as np
import scipy as sp
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: skip
---
import pymc as pm
print(f"Running on PyMC v{pm.__version__}")
import arviz as az
```

```{code-cell} ipython3
import matplotlib.pyplot as plt
%matplotlib inline
plt.rcParams["figure.figsize"] = [12,8]
from matplotlib.patches import Arc, FancyArrowPatch
```

```{code-cell} ipython3
import lighthouse as lh
```

# Lighthouse

+++

In this assignment you will continue to work on the estimatio of the position of the lighthouse, but this time using the Monte-Carlo method.

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
h=1
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
flash_x = np.loadtxt('lighthouse.txt')
```

## Problem 1

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Estimate the position of the lighthouse using PyMC.

1. Formulate the model. When formulating the model please take care that all the parameters of the distributions are specified as floating point numbers (use the decimal point). Specifying them as integers may lead to errors :(
2. Find the MAP estimate.
3. Simulate the posterior and find the mean and 95% highest density interval. The needed flash distribution is called Cauchy. Report the mean and HDI to two decimal places.

+++ {"editable": true, "slideshow": {"slide_type": ""}}

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
with pm.Model() as model_1d:
    mu = pm.Flat('mu')
    obs = pm.Cauchy('obs', alpha=mu, beta=1.0, observed=flash_x)
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
map_1d = pm.find_MAP(model=model_1d)
print(f"MAP: mu = {map_1d['mu']:.2f}")
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
with model_1d:
    trace_1d = pm.sample(2000, return_inferencedata=True)
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
mean_mu = float(trace_1d.posterior['mu'].mean())
hdi_mu = az.hdi(trace_1d, hdi_prob=0.95)['mu'].values
print(f"Mean: {mean_mu:.2f}")
print(f"95% HDI: [{hdi_mu[0]:.2f}, {hdi_mu[1]:.2f}]")
```

## Problem 2

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Estimate the position of the lighthouse and its distance from the shore using PyMC. Use the lighthouse_`2d.txt dataset`. As in previous problem:
1. Formulate the model. When formulating the model please take care that all the parameters of the distributions are specified as floating point numbers (use the decimal point). Specifying them as integers may lead to errors :(
2. Find the MAP estimate.
3. Simulate the posterior and find the mean and 95% highest density interval for each parameter separately.

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
flash_x_2d = np.loadtxt('lighthouse_2d.txt')
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
flash_x_2d[:4]
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
with pm.Model() as model_2d:
    mu = pm.Flat('mu')
    h_lh = pm.HalfFlat('h_lh')
    obs = pm.Cauchy('obs', alpha=mu, beta=h_lh, observed=flash_x_2d)
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
map_2d = pm.find_MAP(model=model_2d)
print(f"MAP: mu = {map_2d['mu']:.2f}, h = {map_2d['h_lh']:.2f}")
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
with model_2d:
    trace_2d = pm.sample(2000, return_inferencedata=True)
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
mean_mu_2d = float(trace_2d.posterior['mu'].mean())
mean_h_2d = float(trace_2d.posterior['h_lh'].mean())
hdi_2d = az.hdi(trace_2d, hdi_prob=0.95)
print(f"mu  - Mean: {mean_mu_2d:.2f}, 95% HDI: [{hdi_2d['mu'].values[0]:.2f}, {hdi_2d['mu'].values[1]:.2f}]")
print(f"h   - Mean: {mean_h_2d:.2f}, 95% HDI: [{hdi_2d['h_lh'].values[0]:.2f}, {hdi_2d['h_lh'].values[1]:.2f}]")
```
