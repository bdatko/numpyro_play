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
# * porting the [importance sampling from pyro](https://pyro.ai/examples/inclined_plane.html) to numpyro
# * pyro importance sampling with a cpuonly PyTorch took forever
# * iterations motivated by the discussion on the [NumPyro forum](https://forum.pyro.ai/t/pyro-example-importance-sampling-port-to-numpyro/3052?u=bdatko)
#
# *Copyright (c) 2017-2019 Uber Technologies, Inc.*
# *SPDX-License-Identifier: Apache-2.0*

# +
import arviz as az
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jax import jit, lax

import numpyro
from numpyro.distributions import Gamma, LogNormal, Normal, Uniform
from numpyro.infer import MCMC, NUTS, SA
# -

# %matplotlib inline
# %reload_ext autoreload
# %autoreload 2
# %load_ext watermark

# %watermark -v -m -p arviz,jax,matplotlib,numpy,pandas,scipy,numpyro

# %watermark -gb

key = jax.random.PRNGKey(2)
key

# **Problem Story** - Samantha really likes physics---but she likes Pyro even more. Instead of using
# calculus to do her physics lab homework (which she could easily do), she's going
# to use bayesian inference. The problem setup is as follows. In lab she observed
# a little box slide down an inclined plane (length of 2 meters and with an incline of
# 30 degrees) 20 times. Each time she measured and recorded the descent time. The timing
# device she used has a known measurement error of 20 milliseconds. Using the observed
# data, she wants to infer the coefficient of friction $\mu$ between the box and the inclined
# plane. She already has (deterministic) python code that can simulate the amount of time
# that it takes the little box to slide down the inclined plane as a function of $\mu$. Using
# Pyro, she can reverse the simulator and infer $\mu$ from the observed descent times.
#
# | ![ppl_inference](https://bookdown.org/robertness/causalml/docs/fig/inference.png) |
# |:--:|
# | **Fig. 1.** The original source of the image is from [Kevin Smith - tutorial: Probabilistic Programming](https://youtu.be/9SEIYh5BCjc?t=894), but the link is from [Robert Ness's Lecture Notes for Causality in Machine Learning](https://bookdown.org/robertness/causalml/docs/tutorial-on-deep-probabilitic-modeling-with-pyro.html)|

little_g = 9.8  # m/s/s
mu0 = 0.12  # actual coefficient of friction in the experiment
time_measurement_sigma = 0.02  # observation noise in seconds (known quantity)


# **Simulator** - the forward simulator, which does numerical integration of the equations of motion in steps of size dt, and optionally includes measurement noise

# +
def _body(info):
    displacement, length, velocity, dt, acceleration, T = info
    displacement += velocity * dt
    velocity += acceleration * dt
    T += dt

    return displacement, length, velocity, dt, acceleration, T


def _conf(info):
    displacement, length, _, _, _, _ = info
    return displacement < length


def slide(displacement, length, velocity, dt, acceleration, T):
    info = (displacement, length, velocity, dt, acceleration, T)
    res = lax.while_loop(_conf, _body, info)
    return res[-1]


# length=2.0, phi=jnp.pi / 6.0, dt=0.005
@jit
def jax_simulate(mu, key, noise_sigma, length, phi, dt):
    T = jnp.zeros(())
    velocity = jnp.zeros(())
    displacement = jnp.zeros(())
    acceleration = (little_g * jnp.sin(phi)) - (little_g * jnp.cos(phi)) * mu

    T = slide(displacement, length, velocity, dt, acceleration, T)

    return T + noise_sigma * jax.random.normal(key, ())


# -

print(
    "First call: ",
    jax_simulate(mu0, key, time_measurement_sigma, 2.0, jnp.pi / 6.0, 0.005),
)
print(
    "Second call: ",
    jax_simulate(0.14, key, time_measurement_sigma, 2.0, jnp.pi / 6.0, 0.005),
)
print(
    "Third call, different type: ",
    jax_simulate(0, key, time_measurement_sigma, 2.0, jnp.pi / 6.0, 0.005),
)


# analytic formula that the simulator above is computing via
# numerical integration (no measurement noise)
@jax.jit
def analytic_T(mu, length=2.0, phi=jnp.pi / 6.0):
    numerator = 2.0 * length
    denominator = little_g * (jnp.sin(phi) - mu * jnp.cos(phi))
    return jnp.sqrt(numerator / denominator)


# generate N_obs observations using simulator and the true coefficient of friction mu0
print("generating simulated data using the true coefficient of friction %.3f" % mu0)
N_obs = 20

keys = jax.random.split(key, N_obs)

observed_data = jnp.array(
    [
        jax_simulate(mu0, key, time_measurement_sigma, 2.0, jnp.pi / 6.0, 0.005)
        for key in keys
    ]
)
observed_mean = jnp.mean(observed_data)
observed_mean

w = lambda info: jax.lax.while_loop(_conf, _body, info)


def numpyro_model(observed_data, measurment_sigma):
    length = 2.0
    phi = jnp.pi / 6.0
    dt = 0.005
    mu = numpyro.sample("mu", Uniform(0.0, 1.0))

    with numpyro.plate("data_loop", len(observed_data)):
        T = jnp.zeros(())
        velocity = jnp.zeros(())
        displacement = jnp.zeros(())
        acceleration = (little_g * jnp.sin(phi)) - (little_g * jnp.cos(phi)) * mu
        info = (displacement, length, velocity, dt, acceleration, T)
        res = jax.lax.cond(
            acceleration <= 0, info, lambda _: (0.0, 0.0, 0.0, 0.0, 0.0, 1.0e5), info, w
        )
        T_simulated = res[-1]
        numpyro.sample("obs", Normal(T_simulated, measurment_sigma), obs=observed_data)

    return mu


numpyro.render_model(
    numpyro_model,
    model_args=(observed_data, time_measurement_sigma),
    render_distributions=True,
)

for depth in (10, 13, 15):
    nuts_kernel = NUTS(
        numpyro_model, forward_mode_differentiation=True, max_tree_depth=depth
    )
    mcmc = MCMC(
        nuts_kernel,
        num_warmup=500,
        num_samples=1000,
        num_chains=4,
        chain_method="parallel",
        progress_bar=True,
    )
    mcmc.run(
        key, observed_data, time_measurement_sigma, extra_fields=("potential_energy",)
    )
    print(f"--" * 25 + f"max_tree_depth={depth}" + f"--" * 25)
    mcmc.print_summary()

ds = az.from_numpyro(mcmc)

inferred_mu = ds.posterior["mu"].mean().item()
inferred_mu_uncertainty = ds.posterior["mu"].std().item()
print(
    "the coefficient of friction inferred by pyro is %.3f +- %.3f"
    % (inferred_mu, inferred_mu_uncertainty)
)

print("the mean observed descent time in the dataset is: %.4f seconds" % observed_mean)
print(
    "the (forward) simulated descent time for the inferred (mean) mu is: %.4f seconds"
    % jax.jit(jax_simulate)(inferred_mu, key, 0.0, 2.0, jnp.pi / 6.0, 0.005)
)
print(
    (
        "disregarding measurement noise, elementary calculus gives the descent time\n"
        + "for the inferred (mean) mu as: %.4f seconds"
    )
    % analytic_T(inferred_mu)
)

az.plot_density(ds.posterior, var_names=["mu"])

az.plot_trace(ds)
plt.show()

az.plot_rank(ds)
plt.show()


# * Trying Sample Adaptive (SA) kernel from the [docs](http://num.pyro.ai/en/stable/examples/baseball.html?highlight=SA#example-baseball-batting-average)
#     > Note that the Sample Adaptive (SA) kernel, which is implemented based on [5], requires large num_warmup and num_samples (e.g. 15,000 and 300,000). So it is better to disable progress bar to avoid dispatching overhead.

# +
def numpyro_model(observed_data, measurment_sigma):
    length = 2.0
    phi = jnp.pi / 6.0
    dt = 0.005
    mu = numpyro.sample("mu", Uniform(0.0, 1.0))

    with numpyro.plate("data_loop", len(observed_data)):
        T = jnp.zeros(())
        velocity = jnp.zeros(())
        displacement = jnp.zeros(())
        acceleration = (little_g * jnp.sin(phi)) - (little_g * jnp.cos(phi)) * mu
        info = (displacement, length, velocity, dt, acceleration, T)
        res = jax.lax.cond(
            acceleration <= 0, info, lambda _: (0.0, 0.0, 0.0, 0.0, 0.0, 1.0e5), info, w
        )
        T_simulated = res[-1]
        numpyro.sample("obs", Normal(T_simulated, measurment_sigma), obs=observed_data)

    return mu


sa_kernel = SA(numpyro_model)
num_samples, num_chains = 500_000, 20
mcmc = MCMC(
    sa_kernel,
    num_warmup=num_samples // 2,
    num_samples=num_samples,
    num_chains=num_chains,
    chain_method="parallel",
    progress_bar=False,
)
mcmc.run(key, observed_data, time_measurement_sigma, extra_fields=("potential_energy",))

for row in mcmc.get_samples()["mu"].reshape((num_chains, num_samples)):
    inferred_mu = row.mean().item()
    inferred_mu_uncertainty = row.std().item()
    print(
        "the coefficient of friction inferred by pyro is %.3f +- %.3f"
        % (inferred_mu, inferred_mu_uncertainty)
    )


mcmc.print_summary()

# +
for row in mcmc.get_samples()["mu"].reshape((num_chains, num_samples)):
    plt.plot(row)

plt.show()
# -

plt.plot(mcmc.get_samples()["mu"])

# **Prior**
#
# From [Wikipedia on PTFE](https://en.wikipedia.org/wiki/Polytetrafluoroethylene):
# *The coefficient of friction of plastics is usually measured against polished steel.[24] PTFE's coefficient of friction is **0.05 to 0.10**,[15] which is the **third-lowest of any known solid material** (aluminium magnesium boride (BAM) being the first, with a coefficient of friction of 0.02; diamond-like carbon being second-lowest at 0.05)*
#
# Also largets value from the table for both static and sliding is **1.4 for Ag** and **3.0 for Pt**, respectfully [Wikipedia on Friction](https://en.wikipedia.org/wiki/Friction)
#
# I guess the school's physics lab could be doing the experiment with tracks made of Pt or Ag.... but I highly doubt

b = Gamma(2, 2).sample(jax.random.PRNGKey(12), (int(1e4),))
az.plot_kde(b)
plt.show()


# +
def numpyro_model(observed_data, measurment_sigma):
    length = 2.0
    phi = jnp.pi / 6.0
    dt = 0.005
    mu = numpyro.sample("mu", Gamma(2, 2))

    with numpyro.plate("data_loop", len(observed_data)):
        T = jnp.zeros(())
        velocity = jnp.zeros(())
        displacement = jnp.zeros(())
        acceleration = (little_g * jnp.sin(phi)) - (little_g * jnp.cos(phi)) * mu
        info = (displacement, length, velocity, dt, acceleration, T)
        res = jax.lax.cond(
            acceleration <= 0, info, lambda _: (0.0, 0.0, 0.0, 0.0, 0.0, 1.0e5), info, w
        )
        T_simulated = res[-1]
        numpyro.sample("obs", Normal(T_simulated, measurment_sigma), obs=observed_data)

    return mu


for depth in (10, 13, 15):
    nuts_kernel = NUTS(
        numpyro_model, forward_mode_differentiation=True, max_tree_depth=depth
    )
    mcmc = MCMC(
        nuts_kernel,
        num_warmup=500,
        num_samples=1000,
        num_chains=4,
        chain_method="parallel",
        progress_bar=True,
    )
    mcmc.run(
        key, observed_data, time_measurement_sigma, extra_fields=("potential_energy",)
    )
    print(f"--" * 25 + f"max_tree_depth={depth}" + f"--" * 25)
    mcmc.print_summary()

ds = az.from_numpyro(mcmc)
inferred_mu = ds.posterior["mu"].mean().item()
inferred_mu_uncertainty = ds.posterior["mu"].std().item()
print(
    "the coefficient of friction inferred by pyro is %.3f +- %.3f"
    % (inferred_mu, inferred_mu_uncertainty)
)
print("the mean observed descent time in the dataset is: %.4f seconds" % observed_mean)
print(
    "the (forward) simulated descent time for the inferred (mean) mu is: %.4f seconds"
    % jax.jit(jax_simulate)(inferred_mu, key, 0.0, 2.0, jnp.pi / 6.0, 0.005)
)
print(
    (
        "disregarding measurement noise, elementary calculus gives the descent time\n"
        + "for the inferred (mean) mu as: %.4f seconds"
    )
    % analytic_T(inferred_mu)
)
# -

az.plot_trace(ds)
plt.show()

az.plot_rank(ds)
plt.show()


# +
def numpyro_model(observed_data, measurment_sigma):
    length = 2.0
    phi = jnp.pi / 6.0
    dt = 0.005
    mu = numpyro.sample("mu", Gamma(2, 2))

    with numpyro.plate("data_loop", len(observed_data)):
        T = jnp.zeros(())
        velocity = jnp.zeros(())
        displacement = jnp.zeros(())
        acceleration = (little_g * jnp.sin(phi)) - (little_g * jnp.cos(phi)) * mu
        info = (displacement, length, velocity, dt, acceleration, T)
        res = jax.lax.cond(
            acceleration <= 0, info, lambda _: (0.0, 0.0, 0.0, 0.0, 0.0, 1.0e5), info, w
        )
        T_simulated = res[-1]
        numpyro.sample("obs", Normal(T_simulated, measurment_sigma), obs=observed_data)

    return mu


sa_kernel = SA(numpyro_model)
num_samples, num_chains = 500_000, 20
mcmc = MCMC(
    sa_kernel,
    num_warmup=num_samples // 2,
    num_samples=num_samples,
    num_chains=num_chains,
    chain_method="parallel",
    progress_bar=False,
)
mcmc.run(key, observed_data, time_measurement_sigma, extra_fields=("potential_energy",))

for row in mcmc.get_samples()["mu"].reshape((num_chains, num_samples)):
    inferred_mu = row.mean().item()
    inferred_mu_uncertainty = row.std().item()
    print(
        "the coefficient of friction inferred by pyro is %.3f +- %.3f"
        % (inferred_mu, inferred_mu_uncertainty)
    )


mcmc.print_summary()

# +
for row in mcmc.get_samples()["mu"].reshape((num_chains, num_samples)):
    plt.plot(row)

plt.show()
# -

plt.plot(mcmc.get_samples()["mu"])
