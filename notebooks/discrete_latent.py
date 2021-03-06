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
# * playing around with `infer_discrete`
# * The example below show itreations based on the discussion on the [Pyro forum](https://forum.pyro.ai/t/mcmc-get-samples-returns-empty-dict/3086)
# * If you need sometihng similar look at [`Predictive`](http://num.pyro.ai/en/latest/utilities.html#predictive) and [Example: Bayesian Models of Annotation](http://num.pyro.ai/en/latest/examples/annotation.html) from [Support infer_discrete for Predictive (#1086) ](https://github.com/pyro-ppl/numpyro/commit/003424bb3c57e44b433991cc73ddbb557bf31f3c)

import jax
import jax.numpy as jnp
import numpyro
from numpyro.contrib.funsor import config_enumerate, infer_discrete
import numpyro.distributions as dist
from numpyro.infer.util import Predictive
import pandas as pd
from numpyro.infer import MCMC, NUTS, DiscreteHMCGibbs

# %matplotlib inline
# %reload_ext autoreload
# %autoreload 2
# %load_ext watermark

# %watermark -v -m -p jax,numpy,pandas,numpyro

# %watermark -gb

num_samples = 1000
num_warmup = 1000
num_chains = 4

# ## DiscreteHMCGibbs

# +
key = jax.random.PRNGKey(2)

guess = 0.7


def mystery(guess):
    weapon_cpt = jnp.array([[0.9, 0.1], [0.2, 0.8]])
    murderer = numpyro.sample("murderer", dist.Bernoulli(guess))
    weapon = numpyro.sample("weapon", dist.Categorical(weapon_cpt[murderer]))
    return murderer, weapon


conditioned_model = numpyro.handlers.condition(mystery, {"weapon": 0.0})

nuts_kernel = NUTS(conditioned_model)

kernel = DiscreteHMCGibbs(nuts_kernel, modified=True)

mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains)
mcmc.run(key, guess)

mcmc.print_summary()

with numpyro.handlers.seed(rng_seed=0):
    samples = []
    for _ in range(1000):
        samples.append(
            tuple(
                [
                    sample.item() if hasattr(sample, "item") else sample
                    for sample in conditioned_model(guess)
                ]
            )
        )

samples = pd.DataFrame(samples, columns=["murderer", "weapon"])

print(pd.crosstab(samples.murderer, samples.weapon, normalize="all"))
# -

# ## `infer_discrete`

num_samples = 1000
num_warmup = 1000
num_chains = 4


# caution: `*data` within infer_discrete_model is a global variable
def infer_discrete_model(rng_key, samples):
    conditioned_model = numpyro.handlers.condition(model, data=samples)
    infer_discrete_model = infer_discrete(
        config_enumerate(conditioned_model), rng_key=rng_key
    )
    with numpyro.handlers.trace() as tr:
        infer_discrete_model(*data)

    return {
        name: site["value"]
        for name, site in tr.items()
        if site["type"] == "sample" and site["infer"].get("enumerate") == "parallel"
    }


# +
def model(guess, weapon):
    weapon_cpt = jnp.array([[0.9, 0.1], [0.2, 0.8]])
    murderer = numpyro.sample("murderer", dist.Bernoulli(guess))
    weapon = numpyro.sample("weapon", dist.Categorical(weapon_cpt[murderer]), obs=weapon)

nuts_kernel = NUTS(model)

data = (guess, 0.)

# caution: HMC will marginalize all the discrete variables, for `model` results in an empty dict from mcmc.get_samples()
mcmc = MCMC(nuts_kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains)
mcmc.run(key, *data)
# -

posterior_samples = mcmc.get_samples()
posterior_samples

num_samples = 4000

discrete_samples = jax.vmap(infer_discrete_model)(
    jax.random.split(jax.random.PRNGKey(1), num_samples), {}
)

discrete_samples["murderer"].mean(), discrete_samples["murderer"].std()

# ## Using Predictive

# +
key = jax.random.PRNGKey(3)

guess = 0.7


def mystery(guess):
    weapon_cpt = jnp.array([[0.9, 0.1], [0.2, 0.8]])
    murderer = numpyro.sample("murderer", dist.Bernoulli(guess))
    weapon = numpyro.sample("weapon", dist.Categorical(weapon_cpt[murderer]))
    return murderer, weapon


conditioned_model = numpyro.handlers.condition(mystery, {"weapon": 0.0})

predictive = Predictive(conditioned_model, num_samples=1000, infer_discrete=True)
samples = predictive(key, guess)
samples["murderer"].mean(), samples["murderer"].std()
# -

