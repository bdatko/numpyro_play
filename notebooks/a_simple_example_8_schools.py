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
# * [Getting started example from the NumPyro docs](http://num.pyro.ai/en/latest/getting_started.html)

# +
import numpy as np
import numpyro

import numpyro.distributions as dist

from jax import random
from numpyro.infer import MCMC, NUTS
# -

# %matplotlib inline
# %reload_ext autoreload
# %autoreload 2
# %load_ext watermark

# %watermark -v -m -p numpy,numpyro,jax

# %watermark -gb

# +
J = 8

y = np.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])

sigma = np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])


# -

# Eight Schools example
def eight_schools(J, sigma, y=None):
    mu = numpyro.sample('mu', dist.Normal(0, 5))
    tau = numpyro.sample('tau', dist.HalfCauchy(5))
    with numpyro.plate('J', J):
        theta = numpyro.sample('theta', dist.Normal(mu, tau))
        numpyro.sample('obs', dist.Normal(theta, sigma), obs=y)


numpyro.render_model(eight_schools, model_args=(J,sigma, y), render_distributions=True)

nuts_kernel = NUTS(eight_schools)

mcmc = MCMC(nuts_kernel, num_warmup=500, num_samples=1000)

rng_key = random.PRNGKey(0)

mcmc.run(rng_key, J, sigma, y=y, extra_fields=('potential_energy',))

mcmc.print_summary()

pe = mcmc.get_extra_fields()['potential_energy']

print('Expected log joint density: {:.2f}'.format(np.mean(-pe)))

from numpyro.infer.reparam import TransformReparam


# Eight Schools example - Non-centered Reparametrization
def eight_schools_noncentered(J, sigma, y=None):
    mu = numpyro.sample('mu', dist.Normal(0, 5))
    tau = numpyro.sample('tau', dist.HalfCauchy(5))
    with numpyro.plate('J', J):
        with numpyro.handlers.reparam(config={'theta': TransformReparam()}):
            theta = numpyro.sample('theta', dist.TransformedDistribution(dist.Normal(0., 1.), dist.transforms.AffineTransform(mu, tau)))
            numpyro.sample('obs', dist.Normal(theta, sigma), obs=y)


numpyro.render_model(eight_schools_noncentered, model_args=(J,sigma, y), render_distributions=True)

nuts_kernel = NUTS(eight_schools_noncentered)

mcmc = MCMC(nuts_kernel, num_warmup=500, num_samples=1000)

rng_key = random.PRNGKey(0)

mcmc.run(rng_key, J, sigma, y=y, extra_fields=('potential_energy',))


mcmc.print_summary(exclude_deterministic=False)

pe = mcmc.get_extra_fields()['potential_energy']

# Compare with the earlier value
print('Expected log joint density: {:.2f}'.format(np.mean(-pe)))

from numpyro.infer import Predictive


def new_school():
    mu = numpyro.sample('mu', dist.Normal(0, 5))
    tau = numpyro.sample('tau', dist.HalfCauchy(5))
    return numpyro.sample('obs', dist.Normal(mu, tau))


predictive = Predictive(new_school, mcmc.get_samples())

samples_predictive = predictive(random.PRNGKey(1))

print(np.mean(samples_predictive['obs']))

import arviz as az

ds = az.from_numpyro(mcmc)

ds.posterior

# +
param_vars = ["theta"]

az.plot_pair(ds, var_names=param_vars, divergences=True)
# -

az.plot_density(ds.posterior, var_names=param_vars)


