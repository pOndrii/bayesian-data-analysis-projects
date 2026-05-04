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

+++ {"editable": true, "slideshow": {"slide_type": ""}}

# Model selection

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
import numpy as np
import scipy
import pymc as pm
import arviz as az
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
print(f"Running on PyMC version {pm.__version__} and ArviZ version {az.__version__}")
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize']=(8,6)
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Your aim in this assignment is to find the model that best fits the data

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
data= np.loadtxt('data.txt')
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

The data is from an discrete distribution. You can use the function below to plot the histogram.

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
def plot_discrete_hist(data, ax = None, **kwargs):
    if ax is None:
        ax =plt.gca()
    min, max = data.min(), data.max()
    n_bins=int(max-min+1)
    return ax.hist(data, bins=n_bins, range=(min-0.5, max+0.5), **kwargs)
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
fig, ax = plt.subplots()
plot_discrete_hist(data, ax=ax);
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

### Problem 1

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Check if the data can be explained by the binomial distribution with $n=35$. Make a posterior predictive check.

```{code-cell} ipython3
tune = 1000
draws = 8000
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
with pm.Model() as model_bin35:
    p_35 = pm.Beta('p_35', alpha=1, beta=1)
    y_35 = pm.Binomial('y_35', n=35, p=p_35, observed=data)
    trace_bin35 = pm.sample(draws=draws, tune=tune, return_inferencedata=True)
    
    pm.compute_log_likelihood(trace_bin35)
    
    pm.sample_posterior_predictive(trace_bin35, extend_inferencedata=True)

az.plot_ppc(trace_bin35)
plt.title('PPC: Binomial n=35')
plt.show()
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

### Problem 2

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Repeat the same for binomial distribution with $n=45$.

```{code-cell} ipython3
with pm.Model() as model_bin45:
    p_45 = pm.Beta('p_45', alpha=1, beta=1)
    y_45 = pm.Binomial('y_45', n=45, p=p_45, observed=data)
    trace_bin45 = pm.sample(draws=draws, tune=tune, return_inferencedata=True)
    
    pm.compute_log_likelihood(trace_bin45)
    
    pm.sample_posterior_predictive(trace_bin45, extend_inferencedata=True)

az.plot_ppc(trace_bin45)
plt.title('PPC: Binomial n=45')
plt.show()
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

### Problem 3

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Repeat the same for poisson distribution.

+++ {"editable": true, "slideshow": {"slide_type": ""}}

$$p(k)=e^{-\lambda}\frac{\lambda^k}{k!}$$

+++ {"editable": true, "slideshow": {"slide_type": ""}}

 Use $\Gamma$ distribution for prior on the $\lambda$ parameter of the distribution.

```{code-cell} ipython3
with pm.Model() as model_pois:
    lam = pm.Gamma('lam', alpha=1, beta=0.1)
    y_pois = pm.Poisson('y_pois', mu=lam, observed=data)
    trace_pois = pm.sample(draws=draws, tune=tune, return_inferencedata=True)
    
    pm.compute_log_likelihood(trace_pois)
    
    pm.sample_posterior_predictive(trace_pois, extend_inferencedata=True)

az.plot_ppc(trace_pois)
plt.title('PPC: Poisson')
plt.show()
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

### Problem 4

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Compare the three models using the "leave one out" cross validation.

+++ {"editable": true, "slideshow": {"slide_type": ""}}

If you can, explain the results :)

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
comp_dict = {
    'Binomial n=35': trace_bin35,
    'Binomial n=45': trace_bin45,
    'Poisson': trace_pois
}

comp = az.compare(comp_dict, ic='loo')
print(comp)

az.plot_compare(comp)
plt.title('Leave-One-Out Cross-Validation Comparison')
plt.show()
```
