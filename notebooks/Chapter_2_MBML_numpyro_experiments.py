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

# +
import operator
from functools import reduce
from typing import List

import arviz as az
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, DiscreteHMCGibbs
from numpyro.infer.util import Predictive

rng_key = jax.random.PRNGKey(2)
# -

expected = pd.DataFrame(
    [
        (False, False, False, 0.101, 0.101),
        (True, False, False, 0.802, 0.034),
        (False, True, False, 0.034, 0.802),
        (True, True, False, 0.561, 0.561),
        (False, False, True, 0.148, 0.148),
        (True, False, True, 0.862, 0.326),
        (False, True, True, 0.326, 0.862),
        (True, True, True, 0.946, 0.946),
    ],
    columns=["IsCorrect1", "IsCorrect2", "IsCorrect2", "P(csharp)", "P(sql)"],
)

# # Purpose
# - Reproducing [`fritzo`'s answer](https://forum.pyro.ai/t/model-based-machine-learning-book-chapter-2-skills-example-in-pyro-tensor-dimension-issue/464/12?u=bdatko) to [Chapter 2 MBML Learning skills](https://mbmlbook.com/LearningSkills.html)
#
# The twist:
# 1. we are using `numpyro.__version__ == 1.7.1` instead of `pyro.__version__ == 0.3`
# 1. assume a fixed guessing probability (work on building one the first iterations of the model from the book)
# 2. reporduce the results for just three questions, two skills using model form [**Figure 2.17**](https://mbmlbook.com/LearningSkills_Moving_to_real_data.html) with [**Table 2.4**](https://mbmlbook.com/LearningSkills_Testing_out_the_model.html), reproduced below
#
# |    | IsCorrect1   | IsCorrect2   | IsCorrect2   |   P(csharp) |   P(sql) |
# |---:|:-------------|:-------------|:-------------|------------:|---------:|
# |  0 | False        | False        | False        |       0.101 |    0.101 |
# |  1 | True         | False        | False        |       0.802 |    0.034 |
# |  2 | False        | True         | False        |       0.034 |    0.802 |
# |  3 | True         | True         | False        |       0.561 |    0.561 |
# |  4 | False        | False        | True         |       0.148 |    0.148 |
# |  5 | True         | False        | True         |       0.862 |    0.326 |
# |  6 | False        | True         | True         |       0.326 |    0.862 |
# |  7 | True         | True         | True         |       0.946 |    0.946 |
#
# The table above can be used to check our model, and to get us ready for the *real data*. Lets view each permutation as a data record, resulting in a table of 3 responses from 8 people, where each question either needs `skill_01`, `skill_02`, or `skill_01` and `skill_02`. The toy data is shown below:

responses_check = jnp.array([[0., 1., 0., 1., 0., 1., 0., 1.], [0., 0., 1., 1., 0., 0., 1., 1.], [0., 0., 0., 0., 1., 1., 1., 1.]])
skills_needed_check = [[0], [1], [0, 1]]


# - I have been playing around with various model and inference engines
# - trying out iterations based on the discussion on the [Pyro forum](https://forum.pyro.ai/t/numpyro-chapter-2-mbml/3184?u=bdatko)

# #### model_00
# * trying out the two for loops over skills
# * beta  priors for skills

def model_00(
    graded_responses, skills_needed: List[List[int]], prob_mistake=0.1, prob_guess=0.2
):
    n_questions, n_participants = graded_responses.shape
    n_skills = max(map(max, skills_needed)) + 1
    
    participants_plate = numpyro.plate("participants_plate", n_participants)
    
    with participants_plate:
        with numpyro.plate("skills_plate", n_skills):
            theta = numpyro.sample("theta", dist.Beta(1,1))
            
    skills = []
            
    for s in range(n_skills):
        skills.append([])
        for p in range(n_participants):
            sample = numpyro.sample("skill_{}_{}".format(s,p), dist.Bernoulli(theta[s,p]))
            skills[s].append(sample.squeeze())
    

    for q in range(n_questions):
        has_skills = reduce(operator.mul, [jnp.array(skills[i]) for i in skills_needed[q]])
        for p in range(n_participants):
            prob_correct = has_skills[p] * (1 - prob_mistake) + (1 - has_skills[p]) * prob_guess
            isCorrect = numpyro.sample("isCorrect_{}_{}".format(q,p), dist.Bernoulli(prob_correct), obs=graded_responses[q,p],)

# +
nuts_kernel = NUTS(model_00)

kernel = DiscreteHMCGibbs(nuts_kernel, modified=True)

mcmc = MCMC(kernel, num_warmup=200, num_samples=1000, num_chains=1)
mcmc.run(rng_key, responses_check, skills_needed_check)
mcmc.print_summary()
# -

expected["model_00 P(csharp)"] = [mcmc.get_samples()[key].mean() for key in list(mcmc.get_samples().keys())[:8]]
expected["model_00 P(sql)"] = [mcmc.get_samples()[key].mean() for key in list(mcmc.get_samples().keys())[8:-1]]


# * below the code results in an AssertionError
# * trying using `infer_discrete` without NUTS and MCMC
# * probably b/c of the beta priors? I am not sure though
#
# ```python
# predictive = Predictive(
#     model_00,
#     num_samples=1000,
#     infer_discrete=True,
# )
# discrete_samples = predictive(rng_key, responses_check, skills_needed_check)
# ```
#
#
# ```python
# AssertionError                            Traceback (most recent call last)
# <ipython-input-5-4a9b79af0b50> in <module>
#       4     infer_discrete=True,
#       5 )
# ----> 6 discrete_samples = predictive(rng_key, responses_check, skills_needed_check)
#
# ~/anaconda3/envs/numpyro_play/lib/python3.8/site-packages/numpyro/infer/util.py in __call__(self, rng_key, *args, **kwargs)
#     892             )
#     893         model = substitute(self.model, self.params)
# --> 894         return _predictive(
#     895             rng_key,
#     896             model,
#
# ~/anaconda3/envs/numpyro_play/lib/python3.8/site-packages/numpyro/infer/util.py in _predictive(rng_key, model, posterior_samples, batch_shape, return_sites, infer_discrete, parallel, model_args, model_kwargs)
#     737     rng_key = rng_key.reshape(batch_shape + (2,))
#     738     chunk_size = num_samples if parallel else 1
# --> 739     return soft_vmap(
#     740         single_prediction, (rng_key, posterior_samples), len(batch_shape), chunk_size
#     741     )
#
# ~/anaconda3/envs/numpyro_play/lib/python3.8/site-packages/numpyro/util.py in soft_vmap(fn, xs, batch_ndims, chunk_size)
#     403         fn = vmap(fn)
#     404 
# --> 405     ys = lax.map(fn, xs) if num_chunks > 1 else fn(xs)
#     406     map_ndims = int(num_chunks > 1) + int(chunk_size > 1)
#     407     ys = tree_map(
#
#     [... skipping hidden 15 frame]
#
# ~/anaconda3/envs/numpyro_play/lib/python3.8/site-packages/numpyro/infer/util.py in single_prediction(val)
#     702             model_trace = prototype_trace
#     703             temperature = 1
# --> 704             pred_samples = _sample_posterior(
#     705                 config_enumerate(condition(model, samples)),
#     706                 first_available_dim,
#
# ~/anaconda3/envs/numpyro_play/lib/python3.8/site-packages/numpyro/contrib/funsor/discrete.py in _sample_posterior(model, first_available_dim, temperature, rng_key, *args, **kwargs)
#      60     with funsor.adjoint.AdjointTape() as tape:
#      61         with block(), enum(first_available_dim=first_available_dim):
# ---> 62             log_prob, model_tr, log_measures = _enum_log_density(
#      63                 model, args, kwargs, {}, sum_op, prod_op
#      64             )
#
# ~/anaconda3/envs/numpyro_play/lib/python3.8/site-packages/numpyro/contrib/funsor/infer_util.py in _enum_log_density(model, model_args, model_kwargs, params, sum_op, prod_op)
#     157     model = substitute(model, data=params)
#     158     with plate_to_enum_plate():
# --> 159         model_trace = packed_trace(model).get_trace(*model_args, **model_kwargs)
#     160     log_factors = []
#     161     time_to_factors = defaultdict(list)  # log prob factors
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
# <ipython-input-3-5597a5a9ba33> in model_00(graded_responses, skills_needed, prob_mistake, prob_guess)
#       9     with participants_plate:
#      10         with numpyro.plate("skills_plate", n_skills):
# ---> 11             theta = numpyro.sample("theta", dist.Beta(1,1))
#      12 
#      13     skills = []
#
# ~/anaconda3/envs/numpyro_play/lib/python3.8/site-packages/numpyro/primitives.py in sample(name, fn, obs, rng_key, sample_shape, infer, obs_mask)
#     157 
#     158     # ...and use apply_stack to send it to the Messengers
# --> 159     msg = apply_stack(initial_msg)
#     160     return msg["value"]
#     161 
#
# ~/anaconda3/envs/numpyro_play/lib/python3.8/site-packages/numpyro/primitives.py in apply_stack(msg)
#      29     if msg["value"] is None:
#      30         if msg["type"] == "sample":
# ---> 31             msg["value"], msg["intermediates"] = msg["fn"](
#      32                 *msg["args"], sample_intermediates=True, **msg["kwargs"]
#      33             )
#
# ~/anaconda3/envs/numpyro_play/lib/python3.8/site-packages/numpyro/distributions/distribution.py in __call__(self, *args, **kwargs)
#     300         sample_intermediates = kwargs.pop("sample_intermediates", False)
#     301         if sample_intermediates:
# --> 302             return self.sample_with_intermediates(key, *args, **kwargs)
#     303         return self.sample(key, *args, **kwargs)
#     304 
#
# ~/anaconda3/envs/numpyro_play/lib/python3.8/site-packages/numpyro/distributions/distribution.py in sample_with_intermediates(self, key, sample_shape)
#     573 
#     574     def sample_with_intermediates(self, key, sample_shape=()):
# --> 575         return self._sample(self.base_dist.sample_with_intermediates, key, sample_shape)
#     576 
#     577     def sample(self, key, sample_shape=()):
#
# ~/anaconda3/envs/numpyro_play/lib/python3.8/site-packages/numpyro/distributions/distribution.py in _sample(self, sample_fn, key, sample_shape)
#     532         batch_shape = expanded_sizes + interstitial_sizes
#     533         # shape = sample_shape + expanded_sizes + interstitial_sizes + base_dist.shape()
# --> 534         samples, intermediates = sample_fn(key, sample_shape=sample_shape + batch_shape)
#     535 
#     536         interstitial_dims = tuple(self._interstitial_sizes.keys())
#
# ~/anaconda3/envs/numpyro_play/lib/python3.8/site-packages/numpyro/distributions/distribution.py in sample_with_intermediates(self, key, sample_shape)
#     259         :rtype: numpy.ndarray
#     260         """
# --> 261         return self.sample(key, sample_shape=sample_shape), []
#     262 
#     263     def log_prob(self, value):
#
# ~/anaconda3/envs/numpyro_play/lib/python3.8/site-packages/numpyro/distributions/continuous.py in sample(self, key, sample_shape)
#      79 
#      80     def sample(self, key, sample_shape=()):
# ---> 81         assert is_prng_key(key)
#      82         return self._dirichlet.sample(key, sample_shape)[..., 0]
#      83 
#
# AssertionError: 
# ```

# #### model_01
# * trying out the two for loops over skills, suggested [here](https://forum.pyro.ai/t/numpyro-chapter-2-mbml/3184/2?u=bdatko) and again [here](https://forum.pyro.ai/t/numpyro-chapter-2-mbml/3184/6?u=bdatko)
# * removing beta  priors for skills, more like the book

def model_01a(
    graded_responses, skills_needed: List[List[int]], prob_mistake=0.1, prob_guess=0.2
):
    n_questions, n_participants = graded_responses.shape
    n_skills = max(map(max, skills_needed)) + 1
    
    participants_plate = numpyro.plate("participants_plate", n_participants)
            
    skills = []
            
    for s in range(n_skills):
        skills.append([])
        for p in range(n_participants):
            sample = numpyro.sample("skill_{}_{}".format(s,p), dist.Bernoulli(0.5))
            skills[s].append(sample.squeeze())
    

    for q in range(n_questions):
        has_skills = reduce(operator.mul, [jnp.array(skills[i]) for i in skills_needed[q]])
        for p in range(n_participants):
            prob_correct = has_skills[p] * (1 - prob_mistake) + (1 - has_skills[p]) * prob_guess
            isCorrect = numpyro.sample("isCorrect_{}_{}".format(q,p), dist.Bernoulli(prob_correct), obs=graded_responses[q,p],)

# +
nuts_kernel = NUTS(model_01a)

kernel = DiscreteHMCGibbs(nuts_kernel, modified=True)

mcmc = MCMC(kernel, num_warmup=200, num_samples=1000, num_chains=1)
mcmc.run(rng_key, responses_check, skills_needed_check)
mcmc.print_summary()
# -

expected["model_01a P(csharp)"] = [mcmc.get_samples()[key].mean() for key in list(mcmc.get_samples().keys())[:8]]
expected["model_01a P(sql)"] = [mcmc.get_samples()[key].mean() for key in list(mcmc.get_samples().keys())[8:]]


def model_01b(
    graded_responses, skills_needed: List[List[int]], prob_mistake=0.1, prob_guess=0.2
):
    n_questions, n_participants = graded_responses.shape
    n_skills = max(map(max, skills_needed)) + 1
    
    participants_plate = numpyro.plate("participants_plate", n_participants)
            
    skills = []
            
    for s in range(n_skills):
        skills.append([])
        for p in range(n_participants):
            sample = numpyro.sample("skill_{}_{}".format(s,p), dist.Bernoulli(0.5))
            skills[s].append(sample.squeeze())
    

    for q in range(n_questions):
        has_skills = reduce(operator.mul, [jnp.array(skills[i]) for i in skills_needed[q]])
        for p in range(n_participants):
            prob_correct = has_skills[p] * (1 - prob_mistake) + (1 - has_skills[p]) * prob_guess
            isCorrect = numpyro.sample("isCorrect_{}_{}".format(q,p), dist.Bernoulli(prob_correct), obs=graded_responses[q,p],)

predictive = Predictive(
    model_01b,
    num_samples=3000,
    infer_discrete=True,
)
discrete_samples = predictive(rng_key, responses_check, skills_needed_check)

expected["model_01b P(csharp)"] = [discrete_samples[key].mean() for key in list(discrete_samples.keys())[24:32]]
expected["model_01b P(sql)"] = [discrete_samples[key].mean() for key in list(discrete_samples.keys())[32:]]


# #### model_02
# * trying not to use the doulbe for loop, so slow
# * beta priors for skills
# * this model is very similar to the original post on the forum from [`fritzo`'s answer](https://forum.pyro.ai/t/model-based-machine-learning-book-chapter-2-skills-example-in-pyro-tensor-dimension-issue/464/12?u=bdatko) 

def model_02(
    graded_responses, skills_needed: List[List[int]], prob_mistake=0.1, prob_guess=0.2
):
    n_questions, n_participants = graded_responses.shape
    n_skills = max(map(max, skills_needed)) + 1
    
    participants_plate = numpyro.plate("participants_plate", n_participants)
    
    with participants_plate:
        with numpyro.plate("skills_plate", n_skills):
            theta = numpyro.sample("theta", dist.Beta(1,1))
    
    with participants_plate:
        skills = []
        for s in range(n_skills):
            skills.append(numpyro.sample("skill_{}".format(s), dist.Bernoulli(theta[s])))

    for q in range(n_questions):
        has_skills = reduce(operator.mul, [skills[i] for i in skills_needed[q]])
        prob_correct = has_skills * (1 - prob_mistake) + (1 - has_skills) * prob_guess
        isCorrect = numpyro.sample(
            "isCorrect_{}".format(q),
            dist.Bernoulli(prob_correct).to_event(1),
            obs=graded_responses[q],
        )


# +
nuts_kernel = NUTS(model_02)

kernel = DiscreteHMCGibbs(nuts_kernel, modified=True)

mcmc = MCMC(kernel, num_warmup=200, num_samples=1000, num_chains=1)
mcmc.run(rng_key, responses_check, skills_needed_check)
mcmc.print_summary()
# -

expected["model_02 P(csharp)"] = mcmc.get_samples(group_by_chain=False)["skill_0"].mean(0)
expected["model_02 P(sql)"] = mcmc.get_samples(group_by_chain=False)["skill_1"].mean(0)


# * below the code results in an AssertionError
# * trying using `infer_discrete` without NUTS and MCMC
# * probably b/c of the beta priors? I am not sure though
#
# ```python
# predictive = Predictive(
#     model_02,
#     num_samples=3000,
#     infer_discrete=True,
# )
# discrete_samples = predictive(rng_key, responses_check, skills_needed_check)
# ```
#
#
# ```python
# AssertionError                            Traceback (most recent call last)
# <ipython-input-31-dd19800d774a> in <module>
#       4     infer_discrete=True,
#       5 )
# ----> 6 discrete_samples = predictive(rng_key, responses_check, skills_needed_check)
#
# ~/anaconda3/envs/numpyro_play/lib/python3.8/site-packages/numpyro/infer/util.py in __call__(self, rng_key, *args, **kwargs)
#     892             )
#     893         model = substitute(self.model, self.params)
# --> 894         return _predictive(
#     895             rng_key,
#     896             model,
#
# ~/anaconda3/envs/numpyro_play/lib/python3.8/site-packages/numpyro/infer/util.py in _predictive(rng_key, model, posterior_samples, batch_shape, return_sites, infer_discrete, parallel, model_args, model_kwargs)
#     737     rng_key = rng_key.reshape(batch_shape + (2,))
#     738     chunk_size = num_samples if parallel else 1
# --> 739     return soft_vmap(
#     740         single_prediction, (rng_key, posterior_samples), len(batch_shape), chunk_size
#     741     )
#
# ~/anaconda3/envs/numpyro_play/lib/python3.8/site-packages/numpyro/util.py in soft_vmap(fn, xs, batch_ndims, chunk_size)
#     403         fn = vmap(fn)
#     404 
# --> 405     ys = lax.map(fn, xs) if num_chunks > 1 else fn(xs)
#     406     map_ndims = int(num_chunks > 1) + int(chunk_size > 1)
#     407     ys = tree_map(
#
#     [... skipping hidden 15 frame]
#
# ~/anaconda3/envs/numpyro_play/lib/python3.8/site-packages/numpyro/infer/util.py in single_prediction(val)
#     702             model_trace = prototype_trace
#     703             temperature = 1
# --> 704             pred_samples = _sample_posterior(
#     705                 config_enumerate(condition(model, samples)),
#     706                 first_available_dim,
#
# ~/anaconda3/envs/numpyro_play/lib/python3.8/site-packages/numpyro/contrib/funsor/discrete.py in _sample_posterior(model, first_available_dim, temperature, rng_key, *args, **kwargs)
#      60     with funsor.adjoint.AdjointTape() as tape:
#      61         with block(), enum(first_available_dim=first_available_dim):
# ---> 62             log_prob, model_tr, log_measures = _enum_log_density(
#      63                 model, args, kwargs, {}, sum_op, prod_op
#      64             )
#
# ~/anaconda3/envs/numpyro_play/lib/python3.8/site-packages/numpyro/contrib/funsor/infer_util.py in _enum_log_density(model, model_args, model_kwargs, params, sum_op, prod_op)
#     157     model = substitute(model, data=params)
#     158     with plate_to_enum_plate():
# --> 159         model_trace = packed_trace(model).get_trace(*model_args, **model_kwargs)
#     160     log_factors = []
#     161     time_to_factors = defaultdict(list)  # log prob factors
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
# <ipython-input-25-40b670a4e6b6> in model_02(graded_responses, skills_needed, prob_mistake, prob_guess)
#       9     with participants_plate:
#      10         with numpyro.plate("skills_plate", n_skills):
# ---> 11             theta = numpyro.sample("theta", dist.Beta(1,1))
#      12 
#      13     with participants_plate:
#
# ~/anaconda3/envs/numpyro_play/lib/python3.8/site-packages/numpyro/primitives.py in sample(name, fn, obs, rng_key, sample_shape, infer, obs_mask)
#     157 
#     158     # ...and use apply_stack to send it to the Messengers
# --> 159     msg = apply_stack(initial_msg)
#     160     return msg["value"]
#     161 
#
# ~/anaconda3/envs/numpyro_play/lib/python3.8/site-packages/numpyro/primitives.py in apply_stack(msg)
#      29     if msg["value"] is None:
#      30         if msg["type"] == "sample":
# ---> 31             msg["value"], msg["intermediates"] = msg["fn"](
#      32                 *msg["args"], sample_intermediates=True, **msg["kwargs"]
#      33             )
#
# ~/anaconda3/envs/numpyro_play/lib/python3.8/site-packages/numpyro/distributions/distribution.py in __call__(self, *args, **kwargs)
#     300         sample_intermediates = kwargs.pop("sample_intermediates", False)
#     301         if sample_intermediates:
# --> 302             return self.sample_with_intermediates(key, *args, **kwargs)
#     303         return self.sample(key, *args, **kwargs)
#     304 
#
# ~/anaconda3/envs/numpyro_play/lib/python3.8/site-packages/numpyro/distributions/distribution.py in sample_with_intermediates(self, key, sample_shape)
#     573 
#     574     def sample_with_intermediates(self, key, sample_shape=()):
# --> 575         return self._sample(self.base_dist.sample_with_intermediates, key, sample_shape)
#     576 
#     577     def sample(self, key, sample_shape=()):
#
# ~/anaconda3/envs/numpyro_play/lib/python3.8/site-packages/numpyro/distributions/distribution.py in _sample(self, sample_fn, key, sample_shape)
#     532         batch_shape = expanded_sizes + interstitial_sizes
#     533         # shape = sample_shape + expanded_sizes + interstitial_sizes + base_dist.shape()
# --> 534         samples, intermediates = sample_fn(key, sample_shape=sample_shape + batch_shape)
#     535 
#     536         interstitial_dims = tuple(self._interstitial_sizes.keys())
#
# ~/anaconda3/envs/numpyro_play/lib/python3.8/site-packages/numpyro/distributions/distribution.py in sample_with_intermediates(self, key, sample_shape)
#     259         :rtype: numpy.ndarray
#     260         """
# --> 261         return self.sample(key, sample_shape=sample_shape), []
#     262 
#     263     def log_prob(self, value):
#
# ~/anaconda3/envs/numpyro_play/lib/python3.8/site-packages/numpyro/distributions/continuous.py in sample(self, key, sample_shape)
#      79 
#      80     def sample(self, key, sample_shape=()):
# ---> 81         assert is_prng_key(key)
#      82         return self._dirichlet.sample(key, sample_shape)[..., 0]
#      83 
#
# AssertionError: 
# ```

# #### model_03
# * trying not to use the doulbe for loop, so slow
# * removing beta priors for skills, most similar to the book
# * this model is very similar to the original post on the forum from [`fritzo`'s answer](https://forum.pyro.ai/t/model-based-machine-learning-book-chapter-2-skills-example-in-pyro-tensor-dimension-issue/464/12?u=bdatko) 

def model_03(
    graded_responses, skills_needed: List[List[int]], prob_mistake=0.1, prob_guess=0.2
):
    n_questions, n_participants = graded_responses.shape
    n_skills = max(map(max, skills_needed)) + 1
    
    participants_plate = numpyro.plate("participants_plate", n_participants)
    
    with participants_plate:
        skills = []
        for s in range(n_skills):
            skills.append(numpyro.sample("skill_{}".format(s), dist.Bernoulli(0.5)))

    for q in range(n_questions):
        has_skills = reduce(operator.mul, [skills[i] for i in skills_needed[q]])
        prob_correct = has_skills * (1 - prob_mistake) + (1 - has_skills) * prob_guess
        isCorrect = numpyro.sample(
            "isCorrect_{}".format(q),
            dist.Bernoulli(prob_correct).to_event(1),
            obs=graded_responses[q],
        )


# +
nuts_kernel = NUTS(model_03)

kernel = DiscreteHMCGibbs(nuts_kernel, modified=True)

mcmc = MCMC(kernel, num_warmup=200, num_samples=1000, num_chains=1)
mcmc.run(rng_key, responses_check, skills_needed_check)
mcmc.print_summary()
# -

expected["model_03 P(csharp)"] = mcmc.get_samples(group_by_chain=False)["skill_0"].mean(0)
expected["model_03 P(sql)"] = mcmc.get_samples(group_by_chain=False)["skill_1"].mean(0)

# #### model_04
# * trying using SVI as suggested [here](https://forum.pyro.ai/t/numpyro-chapter-2-mbml/3184/2?u=bdatko) and again [here](https://forum.pyro.ai/t/numpyro-chapter-2-mbml/3184/5?u=bdatko)
# * removed beta priors for skills, most similar to the book
# * this model is very similar to the original post on the forum from [`fritzo`'s answer](https://forum.pyro.ai/t/model-based-machine-learning-book-chapter-2-skills-example-in-pyro-tensor-dimension-issue/464/12?u=bdatko) 

from numpyro.infer import SVI, TraceGraph_ELBO


# +
def model_04(
    graded_responses, skills_needed: List[List[int]], prob_mistake=0.1, prob_guess=0.2
):
    n_questions, n_participants = graded_responses.shape
    n_skills = max(map(max, skills_needed)) + 1

    with numpyro.plate("participants_plate", n_participants):
        with numpyro.plate("skills_plate", n_skills):
            skills = numpyro.sample(
                "skills", dist.Bernoulli(0.5), infer={"enumerate": "parallel"}
            )

    for q in range(n_questions):
        has_skills = reduce(operator.mul, [skills[i] for i in skills_needed[q]])
        prob_correct = has_skills * (1 - prob_mistake) + (1 - has_skills) * prob_guess
        isCorrect = numpyro.sample(
            "isCorrect_{}".format(q),
            dist.Bernoulli(prob_correct).to_event(1),
            obs=graded_responses[q],
        )


def guide_04(
    graded_responses, skills_needed: List[List[int]], prob_mistake=0.1, prob_guess=0.2
):
    _, n_participants = graded_responses.shape
    n_skills = max(map(max, skills_needed)) + 1

    skill_p = numpyro.param(
        "skill_p",
        0.5 * jnp.ones((n_skills, n_participants)),
        constraint=dist.constraints.unit_interval,
    )

    with numpyro.plate("participants_plate", n_participants):
        with numpyro.plate("skills_plate", n_skills):
            skills = numpyro.sample("skills", dist.Bernoulli(skill_p))

    return skills, skill_p


# +
optimizer = numpyro.optim.Adam(step_size=0.05)

svi = SVI(model_04, guide_04, optimizer, loss=TraceGraph_ELBO())
# -

svi_result = svi.run(rng_key, 10_000, responses_check, skills_needed_check)

# params, state, losses
svi_result.params["skill_p"]

plt.semilogy(np.array(svi_result.losses))

expected["model_04 skill_01 P(csharp)"] = np.array(svi_result.params["skill_p"][0])
expected["model_04 skill_02 P(sql)"] = np.array(svi_result.params["skill_p"][1])

expected

# #### model_05
# * trying explict config_enumerate
# * same as `model_03`

from numpyro.contrib.funsor import config_enumerate


@config_enumerate
def model_05(
    graded_responses, skills_needed: List[List[int]], prob_mistake=0.1, prob_guess=0.2
):
    n_questions, n_participants = graded_responses.shape
    n_skills = max(map(max, skills_needed)) + 1
    
    participants_plate = numpyro.plate("participants_plate", n_participants)
    
    with participants_plate:
        skills = []
        for s in range(n_skills):
            skills.append(numpyro.sample("skill_{}".format(s), dist.Bernoulli(0.5)), infer={"enumerate": "parallel"})

    for q in range(n_questions):
        has_skills = reduce(operator.mul, [skills[i] for i in skills_needed[q]])
        prob_correct = has_skills * (1 - prob_mistake) + (1 - has_skills) * prob_guess
        isCorrect = numpyro.sample(
            "isCorrect_{}".format(q),
            dist.Bernoulli(prob_correct).to_event(1),
            obs=graded_responses[q],
        )


# +
nuts_kernel = NUTS(model_03)

kernel = DiscreteHMCGibbs(nuts_kernel, modified=True)

mcmc = MCMC(kernel, num_warmup=200, num_samples=1000, num_chains=1)
mcmc.run(rng_key, responses_check, skills_needed_check)
mcmc.print_summary()
# -


