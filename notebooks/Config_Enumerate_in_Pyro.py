# -*- coding: utf-8 -*-
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
# * Reproduce the example from [Harald Vöhringer's Config Enumerate in Pyro](https://haraldvohringer.com/blog/config-enumerate-in-pyro/)

# ## Pyro Example

import matplotlib.pyplot as plt
import numpy as np
import pyro
import pyro.distributions as dist
import torch
from tqdm import tqdm


# %matplotlib inline
# %reload_ext autoreload
# %autoreload 2
# %load_ext watermark

# %watermark -v -m -p matplotlib,numpy,pyro,torch,tqdm

# %watermark -gb

def sample(n=1):
    urn = pyro.distributions.Categorical(torch.ones(10) / 10).sample()
    draws = pyro.distributions.Binomial(10, urn / 10).sample((n,))
    return urn, draws


true_urn, draws = sample(10)
print(f"{true_urn=} | {draws=}")


# +
@pyro.infer.config_enumerate
def model(y):
    u = pyro.sample("u", pyro.distributions.Dirichlet(torch.ones(10)))
    with pyro.plate("data", y.shape[0]):
        urn = pyro.sample("urn", pyro.distributions.Categorical(u))
        pyro.sample("obs", pyro.distributions.Binomial(10, urn / 10), obs=y)


guide = pyro.infer.autoguide.AutoDiagonalNormal(pyro.poutine.block(model, expose=["u"]))
# -

loss = pyro.infer.TraceEnum_ELBO(max_plate_nesting=1)

pyro.clear_param_store()
adam = pyro.optim.Adam({"lr": 0.001})
svi = pyro.infer.SVI(model, guide, adam, loss)
num_steps = 10000
losses = []
for _ in tqdm(range(num_steps)):
    loss = svi.step(draws)
    losses.append(loss)

plt.semilogy(losses)
ax = plt.gca()
ax.set(ylabel="ELBO", xlabel="Step")

posterior = pyro.infer.Predictive(model, guide=guide, num_samples=5000)
params = posterior(draws)
posterior_u = params["u"].detach().numpy()

print(f"{true_urn=} | {draws=}")

plt.bar(np.arange(10), posterior_u.mean(0).reshape(-1))
_ = plt.gca().set(
    xlabel="Urn", ylabel="Probability", title="Posterior distribution of u"
)

# ## NumPyro Example

# +
import arviz as az
import jax
import jax.numpy as jnp
import pandas as pd

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
# -

# %watermark -v -m -p arviz,jax,pandas,numpyro

key = jax.random.PRNGKey(0)
key, key_split = jax.random.split(key)
key_split, key_split_split = jax.random.split(key_split)


def sample(n, key):
    urn = dist.Categorical(jnp.ones(10) / 10).sample(key)
    draws = dist.Binomial(10, urn / 10).sample(key, (n,))
    return urn, draws


true_urn, draws = sample(10, key_split_split)
print(f"{true_urn=} | {draws=}")


def model(y):
    u = numpyro.sample("u", dist.Dirichlet(jnp.ones(10)))
    with numpyro.plate("data", y.shape[0]):
        urn = numpyro.sample("urn", dist.Categorical(u))
        numpyro.sample("obs", dist.Binomial(10, urn / 10), obs=y)


numpyro.render_model(model, (draws,), render_distributions=True)

nuts_kernel = NUTS(model)
num_chains, num_samples = 4, 1000
mcmc = MCMC(nuts_kernel, num_warmup=500, num_samples=num_samples, num_chains=num_chains)
mcmc.run(key, draws, extra_fields=("potential_energy",))

mcmc.print_summary()

# +
fig, axes = plt.subplots(5, 2, figsize=(10, 10))

for i, ax in enumerate(axes.flatten()):
    for row in mcmc.get_samples()["u"][:, i].reshape((num_chains, num_samples)):
        ax.set_ylim((-0.3, 1.0))
        ax.set_ylabel("probability")
        ax.set_xlabel("Iiteration")
        ax.plot(row)
        ax.set_title(f"u[{i}]")

fig.tight_layout()
plt.show()

# +
fig, axes = plt.subplots(5, 2, figsize=(10, 10))

for i, ax in enumerate(axes.flatten()):
    for row in mcmc.get_samples()["u"][:, i].reshape((num_chains, num_samples)):
        az.plot_density({"u[{}]".format(i): row}, hdi_prob=1, ax=ax)
        ax.set_xticks(np.arange(0, 1, 0.1))
        ax.grid(True)

fig.tight_layout()
plt.show()

# +
# https://stackoverflow.com/a/39622821/3587374
trace_mean = (
    mcmc.get_samples()["u"]
    .reshape(
        mcmc.get_samples()["u"].shape[0] // num_chains,
        -1,
        mcmc.get_samples()["u"].shape[1],
    )
    .mean(axis=0)
)
trace_std = (
    mcmc.get_samples()["u"]
    .reshape(
        mcmc.get_samples()["u"].shape[0] // num_chains,
        -1,
        mcmc.get_samples()["u"].shape[1],
    )
    .std(axis=0)
)

fig, axes = plt.subplots(2, 2, figsize=(8, 8))

for i, ax in enumerate(axes.flatten()):
    ax.bar(np.arange(10), trace_mean[i].reshape(-1), yerr=trace_std[i].reshape(-1))
    ax.set_ylim((0, 0.5))
    ax.set_ylabel("probability")
    ax.set_xlabel("urn")
    ax.set_title(f"Posterior distribution of u form chain_{i}")

fig.tight_layout()
plt.show()
# -

fig, ax = plt.subplots(figsize=(4, 4))
ax.bar(
    np.arange(10),
    mcmc.get_samples()["u"].mean(axis=0).reshape(-1),
    yerr=mcmc.get_samples()["u"].std(axis=0).reshape(-1),
)
ax.set_ylim((0, 0.5))
ax.set_ylabel("probability")
ax.set_xlabel("urn")
ax.set_title(f"Posterior distribution of u\ntrue urn = {true_urn.item()}")
fig.tight_layout()
plt.show()
