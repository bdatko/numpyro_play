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
# * calculate the `log_probability` of the model as shown [Figure 2.29 from MBML](https://www.mbmlbook.com/LearningSkills_Learning_the_guess_probabilities.html)

# +
import operator
import sys
from functools import reduce
from typing import List, Tuple

import arviz as az
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Latex as lt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle

import numpyro
import numpyro.distributions as dist
from numpyro.contrib.funsor import config_enumerate, enum, infer_discrete
from numpyro.handlers import seed, trace
from numpyro.infer import MCMC, NUTS, DiscreteHMCGibbs, log_likelihood
from numpyro.infer.util import Predictive, log_density, potential_energy
# -

rng_key = jax.random.PRNGKey(2)

# +
skills_key = pd.read_csv(
    "http://www.mbmlbook.com/Downloads/LearningSkills_Real_Data_Experiments-Original-Inputs-Quiz-SkillsQuestionsMask.csv",
    header=None,
)
skills_needed = []
for index, row in skills_key.iterrows():
    skills_needed.append([i for i, x in enumerate(row) if x])

responses = pd.read_csv(
    "http://www.mbmlbook.com/Downloads/LearningSkills_Real_Data_Experiments-Original-Inputs-IsCorrect.csv",
    header=None,
)

responses = responses.astype("int32")


# -

def model_00(
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


# + tags=[]
nuts_kernel = NUTS(model_00)

kernel = DiscreteHMCGibbs(nuts_kernel, modified=True)

mcmc = MCMC(kernel, num_warmup=200, num_samples=1000, num_chains=4)
mcmc.run(rng_key, jnp.array(responses), skills_needed)
mcmc.print_summary()
# -

ds = az.from_numpyro(mcmc)

az.plot_trace(ds);

log_density_model_00 = log_density(model_00, (jnp.array(responses), skills_needed), dict(prob_mistake=0.1, prob_guess=0.2), mcmc.get_samples())

log_density_model_00[0]


def model_02(
    graded_responses, skills_needed: List[List[int]], prob_mistake=0.1,
):
    n_questions, n_participants = graded_responses.shape
    n_skills = max(map(max, skills_needed)) + 1
    
    with numpyro.plate("questions_plate", n_questions):
        prob_guess = numpyro.sample("prob_guess", dist.Beta(2.5, 7.5))

    participants_plate = numpyro.plate("participants_plate", n_participants)

    with participants_plate:
        skills = []
        for s in range(n_skills):
            skills.append(numpyro.sample("skill_{}".format(s), dist.Bernoulli(0.5)))

    for q in range(n_questions):
        has_skills = reduce(operator.mul, [skills[i] for i in skills_needed[q]])
        prob_correct = has_skills * (1 - prob_mistake) + (1 - has_skills) * prob_guess[q]
        isCorrect = numpyro.sample(
            "isCorrect_{}".format(q),
            dist.Bernoulli(prob_correct).to_event(1),
            obs=graded_responses[q],
        )


# +
nuts_kernel = NUTS(model_02)

kernel = DiscreteHMCGibbs(nuts_kernel, modified=True)

mcmc_02 = MCMC(kernel, num_warmup=200, num_samples=1000, num_chains=4)
mcmc_02.run(rng_key, jnp.array(responses), skills_needed)
mcmc_02.print_summary()
# -

ds = az.from_numpyro(mcmc_02)

az.plot_trace(ds);

log_density_model_02 = log_density(model_02, (jnp.array(responses), skills_needed), dict(prob_mistake=0.1), mcmc_02.get_samples())

# +
nuts_kernel = NUTS(model_00)

kernel = DiscreteHMCGibbs(nuts_kernel, modified=True)

mcmc = MCMC(kernel, num_warmup=200, num_samples=1000, num_chains=4)
mcmc.run(rng_key, jnp.array(responses), skills_needed, extra_fields=('potential_energy',))
mcmc.print_summary()
# -


