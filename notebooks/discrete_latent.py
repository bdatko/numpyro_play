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

import jax
import jax.numpy as jnp
import numpyro
from numpyro.contrib.funsor import config_enumerate, infer_discrete
import numpyro.distributions as dist
import pandas as pd
from numpyro.infer import MCMC, NUTS, DiscreteHMCGibbs

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

num_samples = 4000

posterior_samples = mcmc.get_samples()
discrete_samples = jax.vmap(infer_discrete_model)(
    jax.random.split(jax.random.PRNGKey(1), num_samples), posterior_samples
)

posterior_samples

discrete_samples["murderer"].mean(), discrete_samples["murderer"].std()


