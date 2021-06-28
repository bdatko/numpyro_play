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
#
# *Copyright (c) 2017-2019 Uber Technologies, Inc.*
# *SPDX-License-Identifier: Apache-2.0*

# +
import numpy as np
import jax
import jax.numpy as jnp
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az

import numpyro
from numpyro.distributions import Normal, Uniform, LogNormal
from numpyro.infer import MCMC, NUTS
# -

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
    res = jax.lax.while_loop(_conf, _body, info)
    return res[-1]

def jax_simulate(mu, key, noise_sigma, length=2.0, phi=jnp.pi / 6.0, dt=0.005):
    T = jnp.zeros(())
    velocity = jnp.zeros(())
    displacement = jnp.zeros(())
    acceleration = (little_g * jnp.sin(phi)) - (little_g * jnp.cos(phi)) * mu
    
    T = slide(displacement, length, velocity, dt, acceleration, T)

    return T + noise_sigma * jax.random.normal(key, ())

print("First call: ", jax.jit(jax_simulate)(mu0, key, time_measurement_sigma))
print ("Second call: ", jax.jit(jax_simulate)(0.14, key, time_measurement_sigma))
print ("Third call, different type: ", jax.jit(jax_simulate)(0, key, time_measurement_sigma))


# -

# analytic formula that the simulator above is computing via
# numerical integration (no measurement noise)
@jax.jit
def analytic_T(mu, length=2.0, phi=jnp.pi / 6.0):
    numerator = 2.0 * length
    denominator = little_g * (jnp.sin(phi) - mu * jnp.cos(phi))
    return np.sqrt(numerator / denominator)


# generate N_obs observations using simulator and the true coefficient of friction mu0
print("generating simulated data using the true coefficient of friction %.3f" % mu0)
N_obs = 20

keys = jax.random.split(key, N_obs)

observed_data = jnp.array([jax.jit(jax_simulate)(mu0, key, time_measurement_sigma) for key in keys])
observed_mean = jnp.mean(observed_data)
observed_mean

# **Prior**
#
# From [Wikipedia on PTFE](https://en.wikipedia.org/wiki/Polytetrafluoroethylene):
# *The coefficient of friction of plastics is usually measured against polished steel.[24] PTFE's coefficient of friction is 0.05 to 0.10,[15] which is the third-lowest of any known solid material (aluminium magnesium boride (BAM) being the first, with a coefficient of friction of 0.02; diamond-like carbon being second-lowest at 0.05)*
#
# Also largets value from the table, regardless of static or sliding, seems to be ~3.0 for Pt [Wikipedia on Friction](https://en.wikipedia.org/wiki/Friction)
#
# I guess the school's physics lab could be doing the experiment with tracks made of Pt.... but I highly doubt

b = LogNormal(0, 0.5).sample(jax.random.PRNGKey(12), (int(1e4),))
az.plot_kde(b)
plt.show()


def numpyro_model(observed_data, measurment_sigma, length=2.0, phi=jnp.pi / 6.0, dt=0.005):
    mu = numpyro.sample("mu", LogNormal(0, 0.5))
    
    with numpyro.plate("data_loop", len(observed_data)):
        T = jnp.zeros(())
        velocity = jnp.zeros(())
        displacement = jnp.zeros(())
        acceleration = (little_g * jnp.sin(phi)) - (little_g * jnp.cos(phi)) * mu
        T_simulated = slide(displacement, length, velocity, dt, acceleration, T)
        numpyro.sample("obs", Normal(T_simulated, measurment_sigma), obs=observed_data)
        
    return mu


numpyro.render_model(numpyro_model, model_args=(observed_data,time_measurement_sigma), render_distributions=True)

nuts_kernel = NUTS(numpyro_model)

mcmc = MCMC(nuts_kernel, num_warmup=500, num_samples=10, num_chains=1, chain_method='parallel', progress_bar=True)

mcmc.run(key, observed_data, time_measurement_sigma, extra_fields=('potential_energy',))


