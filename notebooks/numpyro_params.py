# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: Python [conda env:numpyro_play]
#     language: python
#     name: conda-env-numpyro_play-py
# ---

# # Purpose
# - trying to guess and understand what `params` argument in various [utilities in numpyro](http://num.pyro.ai/en/stable/utilities.html)
# - looking over the docs it's not clear what params should be and the doc strings are either vague or terse IMO
# - `params` is given a doc string on the lins of :
#     * *dictionary of values for param sites of model/guide* from `Predictive`
#     * *dictionary of current parameter values keyed by site name.* from `log_density`
#     * *Dictionary of arrays keyed by names.* from `transform_fn`
#     * *dictionary of unconstrained values keyed by site names.* from `constrain_fn`
#     * *unconstrained parameters of model.* from `potential_energy`
#
# 1. **log_density**
#     - signature: `log_density(model, model_args, model_kwargs, params)`
# 2. **potential_energy**
#     - signature: `potential_energy(model, model_args, model_kwargs, params, enum=False)`
# 3. **log_likelihood**
#     - signature: `log_likelihood(model, posterior_samples, *args, parallel=False, batch_ndims=1, **kwargs)`
#
# The odd ball above is `log_likelihood` since there isn't a `params` argument, but I *feel* like `posterior_samples` might be a good starting place. Looking over the doc strings for `params` I am also thinking about the connection between *unconstrained* term to the doc string of `z` from [HMCState](http://num.pyro.ai/en/stable/mcmc.html?highlight=hmcstate#numpyro.infer.hmc.HMCState):
#
# > `z` - Python collection representing values (unconstrained samples from the posterior) at latent sites.

# +
import jax.numpy as jnp
import numpy as np
from jax import random
from jax.scipy.special import logsumexp

import numpyro
import numpyro.distributions as dist
from numpyro.distributions import constraints
from numpyro.infer import MCMC, NUTS, SVI, Predictive, Trace_ELBO, log_likelihood
from numpyro.infer.util import log_density, potential_energy


# -

# %matplotlib inline
# %reload_ext autoreload
# %autoreload 2
# %load_ext watermark

# %watermark -v -m -p jax,numpy,numpyro

# %watermark -gb

# ### Example of `params`
# * a simple example is from [Stochastic Variational Inference (SVI)](http://num.pyro.ai/en/stable/svi.html?highlight=ELBO#stochastic-variational-inference-svi) doc example

# +
def model(data):
    f = numpyro.sample("latent_fairness", dist.Beta(10, 10))
    with numpyro.plate("N", data.shape[0]):
        numpyro.sample("obs", dist.Bernoulli(f), obs=data)


def guide(data):
    alpha_q = numpyro.param("alpha_q", 15.0, constraint=constraints.positive)
    beta_q = numpyro.param(
        "beta_q",
        lambda rng_key: random.exponential(rng_key),
        constraint=constraints.positive,
    )
    numpyro.sample("latent_fairness", dist.Beta(alpha_q, beta_q))


data = jnp.concatenate([jnp.ones(6), jnp.zeros(4)])
optimizer = numpyro.optim.Adam(step_size=0.0005)
svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
svi_result = svi.run(random.PRNGKey(0), 2000, data)
# -

params = svi_result.params
params

# ### A Simple Example - 8 Schools
# * from [Getting Started with NumPyro](http://num.pyro.ai/en/stable/getting_started.html#getting-started-with-numpyro)

# +
J = 8

y = np.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])

sigma = np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])


# -

# Eight Schools example
def eight_schools(J, sigma, y=None):
    mu = numpyro.sample("mu", dist.Normal(0, 5))
    tau = numpyro.sample("tau", dist.HalfCauchy(5))
    with numpyro.plate("J", J):
        theta = numpyro.sample("theta", dist.Normal(mu, tau))
        numpyro.sample("obs", dist.Normal(theta, sigma), obs=y)


numpyro.render_model(eight_schools, model_args=(J, sigma, y), render_distributions=True)

nuts_kernel = NUTS(eight_schools)

mcmc = MCMC(nuts_kernel, num_warmup=500, num_samples=1000)

rng_key = random.PRNGKey(0)

mcmc.run(
    rng_key,
    J,
    sigma,
    y=y,
    extra_fields=(
        "i",
        "z",
        "z_grad",
        "potential_energy",
        "energy",
        "r",
        "trajectory_length",
        "num_steps",
        "accept_prob",
        "mean_accept_prob",
        "diverging",
        "adapt_state",
        "rng_key",
    ),
)

mcmc.print_summary()

mcmc.get_extra_fields().keys()

pe = mcmc.get_extra_fields()["potential_energy"]

print("Expected log joint density: {:.2f}".format(np.mean(-pe)))

# + tags=[]
mcmc.get_samples().keys()
# -

mcmc.get_samples()["mu"].mean()

mcmc.get_samples()["tau"].mean()

mcmc.get_samples()["theta"].mean(0)

# ### **log_likelihood** 
#
# signature: `log_likelihood(model, posterior_samples, *args, parallel=False, batch_ndims=1, **kwargs)`
#
# * A good example of `log_likelihood` is from the [Example: Baseball Batting Average](http://num.pyro.ai/en/stable/examples/baseball.html#example-baseball-batting-average)

# + tags=[]
log_likelihood(
    eight_schools, mcmc.get_samples(), J, sigma, y=y,
)
# -

post_loglik = log_likelihood(eight_schools, mcmc.get_samples(), J, sigma, y=y,)["obs"]
exp_log_density = logsumexp(post_loglik, axis=0) - jnp.log(jnp.shape(post_loglik)[0])

exp_log_density, exp_log_density.sum()

# ### **log_density** 
#
# signature: `log_density(model, model_args, model_kwargs, params)`

{
    "mu": mcmc.get_samples()["mu"].mean(),
    "tau": mcmc.get_samples()["tau"].mean(),
    "theta": mcmc.get_samples()["theta"].mean(0),
}

# + tags=[]
log_joint_density, model_trace = log_density(
    eight_schools,
    (J, sigma),
    dict(y=y),
    {
        "mu": mcmc.get_samples()["mu"].mean(),
        "tau": mcmc.get_samples()["tau"].mean(),
        "theta": mcmc.get_samples()["theta"].mean(0),
    },
)
# -

log_joint_density

# + tags=[]
for values in mcmc.get_samples().values():
    print(values.shape)

# + [markdown] tags=[]
# * seems like if you pass `mcmc.get_samples()` always results in a `ValueError: Incompatible shapes for broadcasting`
#
# ```python
# log_joint_density, model_trace = log_density(
#     eight_schools,
#     (J, sigma),
#     dict(y=y),
#     mcmc.get_samples(),
# )
#
# ---------------------------------------------------------------------------
# ValueError                                Traceback (most recent call last)
# <ipython-input-26-1d0d15e69ba2> in <module>
# ----> 1 log_joint_density, model_trace = log_density(
#       2     eight_schools,
#       3     (J, sigma),
#       4     dict(y=y),mcmc.get_samples(),
#       5 )
#
# ~/anaconda3/envs/numpyro_play/lib/python3.8/site-packages/numpyro/infer/util.py in log_density(model, model_args, model_kwargs, params)
#      51     """
#      52     model = substitute(model, data=params)
# ---> 53     model_trace = trace(model).get_trace(*model_args, **model_kwargs)
#      54     log_joint = jnp.zeros(())
#      55     for site in model_trace.values():
#
# ~/anaconda3/envs/numpyro_play/lib/python3.8/site-packages/numpyro/handlers.py in get_trace(self, *args, **kwargs)
#     163         :return: `OrderedDict` containing the execution trace.
#     164         """
# --> 165         self(*args, **kwargs)
#     166         return self.trace
#     167 
#
# ~/anaconda3/envs/numpyro_play/lib/python3.8/site-packages/numpyro/primitives.py in __call__(self, *args, **kwargs)
#      85             return self
#      86         with self:
# ---> 87             return self.fn(*args, **kwargs)
#      88 
#      89 
#
# ~/anaconda3/envs/numpyro_play/lib/python3.8/site-packages/numpyro/primitives.py in __call__(self, *args, **kwargs)
#      85             return self
#      86         with self:
# ---> 87             return self.fn(*args, **kwargs)
#      88 
#      89 
#
# <ipython-input-3-d4380d296306> in eight_schools(J, sigma, y)
#       4     tau = numpyro.sample("tau", dist.HalfCauchy(5))
#       5     with numpyro.plate("J", J):
# ----> 6         theta = numpyro.sample("theta", dist.Normal(mu, tau))
#       7         numpyro.sample("obs", dist.Normal(theta, sigma), obs=y)
#
# ~/anaconda3/envs/numpyro_play/lib/python3.8/site-packages/numpyro/primitives.py in sample(name, fn, obs, rng_key, sample_shape, infer, obs_mask)
#     157 
#     158     # ...and use apply_stack to send it to the Messengers
# --> 159     msg = apply_stack(initial_msg)
#     160     return msg["value"]
#     161 
#
# ~/anaconda3/envs/numpyro_play/lib/python3.8/site-packages/numpyro/primitives.py in apply_stack(msg)
#      22     pointer = 0
#      23     for pointer, handler in enumerate(reversed(_PYRO_STACK)):
# ---> 24         handler.process_message(msg)
#      25         # When a Messenger sets the "stop" field of a message,
#      26         # it prevents any Messengers above it on the stack from being applied.
#
# ~/anaconda3/envs/numpyro_play/lib/python3.8/site-packages/numpyro/primitives.py in process_message(self, msg)
#     478             overlap_idx = max(len(expected_shape) - len(dist_batch_shape), 0)
#     479             trailing_shape = expected_shape[overlap_idx:]
# --> 480             broadcast_shape = lax.broadcast_shapes(
#     481                 trailing_shape, tuple(dist_batch_shape)
#     482             )
#
# ~/anaconda3/envs/numpyro_play/lib/python3.8/site-packages/jax/_src/util.py in wrapper(*args, **kwargs)
#     184         return f(*args, **kwargs)
#     185       else:
# --> 186         return cached(config._trace_context(), *args, **kwargs)
#     187 
#     188     wrapper.cache_clear = cached.cache_clear
#
# ~/anaconda3/envs/numpyro_play/lib/python3.8/site-packages/jax/_src/util.py in cached(_, *args, **kwargs)
#     177     @functools.lru_cache(max_size)
#     178     def cached(_, *args, **kwargs):
# --> 179       return f(*args, **kwargs)
#     180 
#     181     @functools.wraps(f)
#
# ~/anaconda3/envs/numpyro_play/lib/python3.8/site-packages/jax/_src/lax/lax.py in broadcast_shapes(*shapes)
#      90   result_shape = _try_broadcast_shapes(shapes)
#      91   if result_shape is None:
# ---> 92     raise ValueError("Incompatible shapes for broadcasting: {}"
#      93                      .format(tuple(map(tuple, shapes))))
#      94   return result_shape
#
# ValueError: Incompatible shapes for broadcasting: ((8,), (1000,))
# ```
# -

# ### **potential_energy** 
#
# signature: `potential_energy(model, model_args, model_kwargs, params, enum=False)`

# + tags=[]
pe_given_unconstrained_params = potential_energy(
    eight_schools,
    (J, sigma),
    dict(y=y),
    {
        "mu": mcmc.get_samples()["mu"].mean(),
        "tau": mcmc.get_samples()["tau"].mean(),
        "theta": mcmc.get_samples()["theta"].mean(0),
    },
)
# -

pe_given_unconstrained_params

# + [markdown] tags=[]
# ```python
# pe_given_unconstrained_params = potential_energy(
#     eight_schools,
#     (J, sigma),
#     dict(y=y),
#     mcmc.get_samples(),
# )
#
# ---------------------------------------------------------------------------
# ValueError                                Traceback (most recent call last)
# <ipython-input-31-2604c02b276c> in <module>
# ----> 1 pe_given_unconstrained_params = potential_energy(
#       2     eight_schools,
#       3     (J, sigma),
#       4     dict(y=y),
#       5     mcmc.get_samples(),
#
# ~/anaconda3/envs/numpyro_play/lib/python3.8/site-packages/numpyro/infer/util.py in potential_energy(model, model_args, model_kwargs, params, enum)
#     225     )
#     226     # no param is needed for log_density computation because we already substitute
# --> 227     log_joint, model_trace = log_density_(
#     228         substituted_model, model_args, model_kwargs, {}
#     229     )
#
# ~/anaconda3/envs/numpyro_play/lib/python3.8/site-packages/numpyro/infer/util.py in log_density(model, model_args, model_kwargs, params)
#      51     """
#      52     model = substitute(model, data=params)
# ---> 53     model_trace = trace(model).get_trace(*model_args, **model_kwargs)
#      54     log_joint = jnp.zeros(())
#      55     for site in model_trace.values():
#
# ~/anaconda3/envs/numpyro_play/lib/python3.8/site-packages/numpyro/handlers.py in get_trace(self, *args, **kwargs)
#     163         :return: `OrderedDict` containing the execution trace.
#     164         """
# --> 165         self(*args, **kwargs)
#     166         return self.trace
#     167 
#
# ~/anaconda3/envs/numpyro_play/lib/python3.8/site-packages/numpyro/primitives.py in __call__(self, *args, **kwargs)
#      85             return self
#      86         with self:
# ---> 87             return self.fn(*args, **kwargs)
#      88 
#      89 
#
# ~/anaconda3/envs/numpyro_play/lib/python3.8/site-packages/numpyro/primitives.py in __call__(self, *args, **kwargs)
#      85             return self
#      86         with self:
# ---> 87             return self.fn(*args, **kwargs)
#      88 
#      89 
#
# ~/anaconda3/envs/numpyro_play/lib/python3.8/site-packages/numpyro/primitives.py in __call__(self, *args, **kwargs)
#      85             return self
#      86         with self:
# ---> 87             return self.fn(*args, **kwargs)
#      88 
#      89 
#
# <ipython-input-3-d4380d296306> in eight_schools(J, sigma, y)
#       4     tau = numpyro.sample("tau", dist.HalfCauchy(5))
#       5     with numpyro.plate("J", J):
# ----> 6         theta = numpyro.sample("theta", dist.Normal(mu, tau))
#       7         numpyro.sample("obs", dist.Normal(theta, sigma), obs=y)
#
# ~/anaconda3/envs/numpyro_play/lib/python3.8/site-packages/numpyro/primitives.py in sample(name, fn, obs, rng_key, sample_shape, infer, obs_mask)
#     157 
#     158     # ...and use apply_stack to send it to the Messengers
# --> 159     msg = apply_stack(initial_msg)
#     160     return msg["value"]
#     161 
#
# ~/anaconda3/envs/numpyro_play/lib/python3.8/site-packages/numpyro/primitives.py in apply_stack(msg)
#      22     pointer = 0
#      23     for pointer, handler in enumerate(reversed(_PYRO_STACK)):
# ---> 24         handler.process_message(msg)
#      25         # When a Messenger sets the "stop" field of a message,
#      26         # it prevents any Messengers above it on the stack from being applied.
#
# ~/anaconda3/envs/numpyro_play/lib/python3.8/site-packages/numpyro/primitives.py in process_message(self, msg)
#     478             overlap_idx = max(len(expected_shape) - len(dist_batch_shape), 0)
#     479             trailing_shape = expected_shape[overlap_idx:]
# --> 480             broadcast_shape = lax.broadcast_shapes(
#     481                 trailing_shape, tuple(dist_batch_shape)
#     482             )
#
# ~/anaconda3/envs/numpyro_play/lib/python3.8/site-packages/jax/_src/util.py in wrapper(*args, **kwargs)
#     184         return f(*args, **kwargs)
#     185       else:
# --> 186         return cached(config._trace_context(), *args, **kwargs)
#     187 
#     188     wrapper.cache_clear = cached.cache_clear
#
# ~/anaconda3/envs/numpyro_play/lib/python3.8/site-packages/jax/_src/util.py in cached(_, *args, **kwargs)
#     177     @functools.lru_cache(max_size)
#     178     def cached(_, *args, **kwargs):
# --> 179       return f(*args, **kwargs)
#     180 
#     181     @functools.wraps(f)
#
# ~/anaconda3/envs/numpyro_play/lib/python3.8/site-packages/jax/_src/lax/lax.py in broadcast_shapes(*shapes)
#      90   result_shape = _try_broadcast_shapes(shapes)
#      91   if result_shape is None:
# ---> 92     raise ValueError("Incompatible shapes for broadcasting: {}"
#      93                      .format(tuple(map(tuple, shapes))))
#      94   return result_shape
#
# ValueError: Incompatible shapes for broadcasting: ((8,), (1000,))
# ```
