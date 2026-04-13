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
import time
import numpy as np
import scipy
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
import matplotlib.pyplot as plt
%matplotlib inline
plt.rcParams["figure.figsize"] = [12,8]
from matplotlib.patches import Arc, FancyArrowPatch

from scipy.special import logsumexp
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

# Lighthouse

+++ {"editable": true, "slideshow": {"slide_type": ""}}

This problem is taken from "Data Analysis, a Bayesian Tutorial" by D.S. Silva with J. Skiling. A lighthouse distance $h=1$ from the shore is rotating with constant angular frequency and emitting thin beams of light at random. The probability of emission is uniform in time. The signals are picked up on the shore by an array of detectors and their location is saved in the file `lighthouse.txt`.  Both the  horizontal location of the lighthouse $x_{lh}$ and its distance from the shore $h$ are unknown. The task is to estimate those positions.

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
flash_x = np.loadtxt('lighthouse_2d.txt')
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

## Sampling distribution

+++ {"editable": true, "slideshow": {"slide_type": ""}}

As we have already established the sampling distribution $P(x|x_{lh},h)$ is given by the Cauchy distribution

+++ {"editable": true, "slideshow": {"slide_type": ""}}

$$P(x|x_{lh},h)=\frac{h}{\pi}\frac{1}{\left(x-x_{lh}\right)^2+h^2}$$

+++ {"editable": true, "slideshow": {"slide_type": ""}}

## Problem 1

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Write down function that implements this probability distribution function (pdf)

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
def p(x_lh, h, x):
    return (h / np.pi) * (1 / ((x - x_lh)**2 + h**2))
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

and the logarithm of the pdf

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
def log_p(x_lh, h, x):
    return np.log(h) - np.log(np.pi) - np.log((x - x_lh)**2 + h**2)
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

## Problem 2

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Calculate the posterior distribution on $x_{lh}$ and $h$ after detection of one flash.
Assume a uniform prior on some interval for both $x_{lh}$ and $h$. Choose the intervals yourself.

Calculate the MAP estimate for $x_{lh}$ and $h$. Then calculate the marginal distributions for $x_{lh}$ and $h$ and plot them. Calculate the MAP estimates for each distributions separately.

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
x_min = -20; x_max=20;
h_min = 1.0; h_max= 10;
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

The distribution will be represented as an array with columns corresponding to the values of $x_{lh}$ and rows to values of $h$. Start by discretizing the intervals you have chosen for $x_{lh}$ and $h$. Use the `np.linspace` function for that

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
nx = 4000
ny = 3000
x_lhs = np.linspace(x_min, x_max, nx)
hs = np.linspace(h_min, h_max, ny)
```

I suggest to use different values for each dimensions. In that way if you by mistake exchange row and columns you will get an error. The prior will be a constant and we do not have to worry about the normalization (we will normalize the posterior) so we will set this value to one

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
prior_array = np.ones((ny,nx))
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

The next step will be to calculate the value of $p(x_0|x_{lh},h)$ for each value $x_{ls}$ and $h$ in arrays `x_lhs` and `hs`, where $x_0$ is the the position of the first flash `flash_x[0]`. This could be done by a simple double loop

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
p_array = np.empty((ny,nx))
p_array.shape
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
start_time = time.time()
for i, x_lh in enumerate(x_lhs):
  for j, h in enumerate(hs):
    p_array[j,i]=p(x_lh,h,flash_x[0])
end_time = time.time()
elapsed_time_loop = end_time-start_time
print(f"Elapsed time = {elapsed_time_loop:.1f}s")
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

However this is not recommended, as the explicit loops in Python tend to be slow. The proper way is to construct a grid of $x_{lh}$ and $h$ values using function [`numpy.meshgrid`](https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html).

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
start_time=time.time()
x_lh_grid, h_grid = np.meshgrid(x_lhs,hs)
end_time = time.time()
elapsed_time_meshgrid = end_time-start_time
print(f"Elapsed time = {elapsed_time_meshgrid:.3f}s")
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Array `x_lh_grid` contains the values of $x_{lh}$ for every cell of array `p_array`. It consists of identical rows

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
x_lh_grid[0]
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
x_lh_grid[1]
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Similarly the array `h_grid` contains the values of $h$ for every cell in array `p_array` and consists of identical columns

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
h_grid[:,0]
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
h_grid[:,1]
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Now we can compute the `p_array` with a single call to `p`

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
start_time=time.time()
p_array = p(x_lh_grid, h_grid, flash_x[0])
end_time = time.time()
elapsed_time_meshgrid_p = end_time-start_time
print(f"Elapsed time = {elapsed_time_meshgrid_p:.3f}s")
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

The whole computation took

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
print(f"{(elapsed_time_meshgrid+elapsed_time_meshgrid_p)*1000:.0f}ms")
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

compared to

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
print(f"{(elapsed_time_loop):.1f}s")
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

for the explicit loop, so the speedup is

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
print(f"{elapsed_time_loop/(elapsed_time_meshgrid+elapsed_time_meshgrid_p):.1f}")
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

times. This is not very important here, as four seconds is still a very short time, but this may quickly become an issue when we add more data.

+++ {"editable": true, "slideshow": {"slide_type": ""}}

The array can be visualized using the `imshow` function, but first we will normalize  the posterior so the sum (not integral) is equal to one.

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
posterior1 = p_array*prior_array
Z = posterior1.sum()
posterior1/=Z
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
plt.imshow(posterior1, origin='lower', extent=(x_min, x_max, h_min, h_max), aspect='auto');
plt.colorbar();
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

but we can get a clearer picture using the `contour` and `contourf` functions

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
levels=20
cs = plt.contourf(x_lhs,hs, posterior1, levels=levels);
plt.contour(x_lhs,hs, posterior1, colors='black', linewidths=0.5, levels=levels);
plt.colorbar(cs);
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Depending on the interval that you have chosen for the prior, you will probably not see  anything on the plots. This is OK as for one flash the distribution is strongly peaked around $(x_0,0)$. This will get better when we obtain more data.

+++ {"editable": true, "slideshow": {"slide_type": ""}}

#### MAP estimate

+++ {"editable": true, "slideshow": {"slide_type": ""}}

__Hint:__ Use the `numpy.argmax` and `numpy.unravel_index` functions.

```{code-cell} ipython3
map_idx = np.unravel_index(np.argmax(posterior1), posterior1.shape)
map_h = hs[map_idx[0]]
map_x = x_lhs[map_idx[1]]
print(f"Joint MAP: x_lh = {map_x}, h = {map_h}")
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

### Marginal distributions

+++ {"editable": true, "slideshow": {"slide_type": ""}}

You can obtain the marginal distribution $p(x_{lh}|x_0)$ by summing the array along the axis zero.

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Similarly the distribution $p(h|x_0)$ will be obtained by summing  over the axis one.

```{code-cell} ipython3
marg_x = posterior1.sum(axis=0)
marg_h = posterior1.sum(axis=1)

map_marg_x = x_lhs[np.argmax(marg_x)]
map_marg_h = hs[np.argmax(marg_h)]
print(f"Marginal MAP: x_lh = {map_marg_x}, h = {map_marg_h}")

plt.plot(x_lhs, marg_x)
plt.xlabel('x_lh')
plt.ylabel('Marginal Probability')
plt.show()

plt.plot(hs, marg_h)
plt.xlabel('h')
plt.ylabel('Marginal Probability')
plt.show()
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

### Logarithms

+++ {"editable": true, "slideshow": {"slide_type": ""}}

More often than not we will be using logarithm of probabilities. Perform same calculations as previously, using the log of the probability.

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
log_prior_array = np.zeros_like(prior_array)
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
log_p_array = log_p(x_lh_grid, h_grid, flash_x[0])
log_posterior1= log_p_array+log_prior_array
Z= logsumexp(log_posterior1)
log_posterior1 -= Z
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
max_log_posterior = log_posterior1.max()
cs = plt.contourf(x_lhs,hs, log_posterior1-max_log_posterior);
plt.contour(x_lhs,hs, log_posterior1-max_log_posterior,colors = 'black', linestyles='-');
plt.colorbar(cs);
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Because logarithm is a monotonically increasing function we can calculate the MAP estimate in the same way as for the  probabilities.

+++ {"editable": true, "slideshow": {"slide_type": ""}}

#### Marginal distributions

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Again we have to partially sum the `log_posterior1` array. But before doing that we have to exponentiate it. To avoid numerical problems it is best to use the `scipy.special.logsumexp` function.

```{code-cell} ipython3
map_log_idx = np.unravel_index(np.argmax(log_posterior1), log_posterior1.shape)
print(f"Log Joint MAP: x_lh = {x_lhs[map_log_idx[1]]}, h = {hs[map_log_idx[0]]}")

log_marg_x = logsumexp(log_posterior1, axis=0)
log_marg_h = logsumexp(log_posterior1, axis=1)

print(f"Log Marginal MAP: x_lh = {x_lhs[np.argmax(log_marg_x)]}")
print(f"Log Marginal MAP: h = {hs[np.argmax(log_marg_h)]}")
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

## Problem 3.1

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Write down a function to compute an array containing the log of the sampling distribution using an array of flash positions, not just single point.

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
def log_p_many(x_lh, h_lh, flash):
    log_prob = np.zeros_like(x_lh)
    for f in flash:
        log_prob += np.log(h_lh) - np.log(np.pi) - np.log((f - x_lh)**2 + h_lh**2)
    return log_prob
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

## Problem 3.2

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Plot the (log)posterior distribution after two flashes. Calculate the MAP estimate, as well as the marginal distributions and MAP estimates for each variable separately.

```{code-cell} ipython3
log_p_array2 = log_p_many(x_lh_grid, h_grid, flash_x[:2])
log_posterior2 = log_p_array2 + log_prior_array
log_posterior2 -= logsumexp(log_posterior2)

map_idx2 = np.unravel_index(np.argmax(log_posterior2), log_posterior2.shape)
print(f"Joint MAP (2 flashes): x_lh = {x_lhs[map_idx2[1]]}, h = {hs[map_idx2[0]]}")

log_marg_x2 = logsumexp(log_posterior2, axis=0)
log_marg_h2 = logsumexp(log_posterior2, axis=1)

print(f"Marginal MAP x_lh: {x_lhs[np.argmax(log_marg_x2)]}")
print(f"Marginal MAP h: {hs[np.argmax(log_marg_h2)]}")

levels = 20
max_log_posterior2 = log_posterior2.max()
cs = plt.contourf(x_lhs, hs, log_posterior2 - max_log_posterior2, levels=levels)
plt.contour(x_lhs, hs, log_posterior2 - max_log_posterior2, colors='black', linewidths=0.5, levels=levels)
plt.colorbar(cs)
plt.xlabel('x_lh')
plt.ylabel('h')
plt.show()

plt.plot(x_lhs, np.exp(log_marg_x2))
plt.xlabel('x_lh')
plt.ylabel('Marginal Probability')
plt.show()

plt.plot(hs, np.exp(log_marg_h2))
plt.xlabel('h')
plt.ylabel('Marginal Probability')
plt.show()
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

## Problem 3.3

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Perform same calculations as in the previous problem, but use all the flash positions.

```{code-cell} ipython3
log_p_array_all = log_p_many(x_lh_grid, h_grid, flash_x)
log_posterior_all = log_p_array_all + log_prior_array
log_posterior_all -= logsumexp(log_posterior_all)

map_idx_all = np.unravel_index(np.argmax(log_posterior_all), log_posterior_all.shape)
print(f"Joint MAP (all flashes): x_lh = {x_lhs[map_idx_all[1]]}, h = {hs[map_idx_all[0]]}")

log_marg_x_all = logsumexp(log_posterior_all, axis=0)
log_marg_h_all = logsumexp(log_posterior_all, axis=1)

print(f"Marginal MAP x_lh: {x_lhs[np.argmax(log_marg_x_all)]}")
print(f"Marginal MAP h: {hs[np.argmax(log_marg_h_all)]}")

levels = 20
max_log_posterior_all = log_posterior_all.max()
cs = plt.contourf(x_lhs, hs, log_posterior_all - max_log_posterior_all, levels=levels)
plt.contour(x_lhs, hs, log_posterior_all - max_log_posterior_all, colors='black', linewidths=0.5, levels=levels)
plt.colorbar(cs)
plt.xlabel('x_lh')
plt.ylabel('h')
plt.show()

plt.plot(x_lhs, np.exp(log_marg_x_all))
plt.xlabel('x_lh')
plt.ylabel('Marginal Probability')
plt.show()

plt.plot(hs, np.exp(log_marg_h_all))
plt.xlabel('h')
plt.ylabel('Marginal Probability')
plt.show()
```
