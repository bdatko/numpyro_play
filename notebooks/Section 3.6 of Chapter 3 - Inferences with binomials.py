# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: Python [conda env:numpyro_play]
#     language: python
#     name: conda-env-numpyro_play-py
# ---

# # Purpose
# * Want to reproduce the summary stats from [PyMC3 3.6 Joint distributions](https://nbviewer.jupyter.org/github/pymc-devs/resources/blob/master/BCM/ParameterEstimation/Binomial.ipynb#3.6-Joint-distributions) from the [book Bayesian Cognitive Modeling](https://bayesmodels.com/)
# * the example was inspired from the post by [yongduek on the Pyro fourm](https://forum.pyro.ai/t/latent-categorical/2695/6?u=bdatko)
# * If you need sometihng similar look at [Predictive](http://num.pyro.ai/en/latest/utilities.html#predictive) and [Example: Bayesian Models of Annotation](http://num.pyro.ai/en/latest/examples/annotation.html) from commit [Support infer_discrete for Predictive (#1086)](https://github.com/pyro-ppl/numpyro/commit/003424bb3c57e44b433991cc73ddbb557bf31f3c)

# +
import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, DiscreteHMCGibbs
from numpyro.infer.util import Predictive
# -

# %matplotlib inline
# %reload_ext autoreload
# %autoreload 2
# %load_ext watermark

# %watermark -v -m -p arviz,jax,numpy,pandas,numpyro

# %watermark -gb

# +
rng_key = jax.random.PRNGKey(2)
keys = jax.random.split(rng_key, 4)

num_warmup = 1000
num_samples = 5000
num_chains = 4


# -

# ### Discrete `n`

# +
def model36(a, b, probs, k=None):
    # priors: th, n
    n = numpyro.sample("n", dist.Categorical(probs=probs))
    th = numpyro.sample("th", dist.Beta(a, b))
    # observation
    size = len(k)
    with numpyro.plate(f"i=1..{size}", size=size):
        obs = numpyro.sample("k", dist.Binomial(total_count=n, probs=th), obs=k)


a, b = 1, 1
k = jnp.array([16, 18, 22, 25, 27])
nmax = 500
probs = jnp.array([1.0] * nmax) / nmax
# -

numpyro.render_model(model36, (a, b, probs, k), render_distributions=True)

kernel = numpyro.infer.DiscreteHMCGibbs(NUTS(model36), modified=True)
mcmc = MCMC(
    kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains
)
mcmc.run(keys[0], a=b, b=b, probs=probs, k=k)
mcmc.print_summary()

ds = az.from_numpyro(mcmc)
az.plot_trace(ds);

# ### Discrete `n`
# #### Using `Predictive`

from numpyro.infer.util import Predictive

kernel = NUTS(model36)
mcmc = MCMC(
    kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains
)
mcmc.run(keys[1], a=b, b=b, probs=probs, k=k)
mcmc.print_summary()

az.plot_trace(mcmc.get_samples());

posterior_samples = mcmc.get_samples()
predictive = Predictive(
    model36, posterior_samples, infer_discrete=True, batch_ndims=num_chains
)
discrete_samples = predictive(keys[2], *(a, b, probs, k))

az.stats.summary(
    np.array(discrete_samples["n"].reshape(num_chains, num_samples)), hdi_prob=0.9
)

az.plot_trace(np.array(discrete_samples["n"].reshape(num_chains, num_samples)));


# ### Continous `n`
# Quote below is from [PyMC3 notebook Chapter 3 - Inferences with binomials - Section: Note from Junpeng Lao](https://nbviewer.jupyter.org/github/pymc-devs/resources/blob/master/BCM/ParameterEstimation/Binomial.ipynb#Note-from-Junpeng-Lao)
# > Actually, we don't necessary need to use DiscreteUniform for TotalN, as the computation of logp in Binomial doesn't require n to be an integer.

def modelu(a, b, nmax, k=None):
    u = numpyro.sample("u", dist.Uniform())
    n = u * nmax
    numpyro.deterministic("n", n)
    th = numpyro.sample("th", dist.Beta(a, b))
    size = len(k)
    with numpyro.plate(f"i=1..{size}", size=size):
        obs = numpyro.sample("k", dist.Binomial(total_count=n, probs=th), obs=k)


# I guess the render doesn't track deterministic values...
numpyro.render_model(modelu, (a, b, nmax, k), render_distributions=True)

# + tags=[]
kernel = NUTS(modelu)
mcmc = MCMC(
    kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains
)
mcmc.run(keys[3], a=b, b=b, nmax=nmax, k=k)
mcmc.print_summary()
# -

ds = az.from_numpyro(mcmc)

az.stats.summary(ds["posterior"]["n"], hdi_prob=0.9)

az.plot_trace(ds);
