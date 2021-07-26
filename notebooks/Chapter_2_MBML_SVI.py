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

# +
import operator
from functools import reduce
from typing import List

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyro
import pyro.distributions as dist
import torch
from pyro import poutine
from pyro.infer import SVI, Trace_ELBO, TraceEnum_ELBO, config_enumerate, infer_discrete
from pyro.ops.indexing import Vindex
from torch.distributions import constraints

pyro.set_rng_seed(0)
# -

# # Purpose
# - Reproducing [`fritzo`'s answer](https://forum.pyro.ai/t/model-based-machine-learning-book-chapter-2-skills-example-in-pyro-tensor-dimension-issue/464/12?u=bdatko) to [Chapter 2 MBML Learning skills](https://mbmlbook.com/LearningSkills.html)
#
# The twist:
# 1. we are using `pyro.__version__ == 1.7.0` instead of `0.3`
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

responses_check = torch.tensor([[0., 1., 0., 1., 0., 1., 0., 1.], [0., 0., 1., 1., 0., 0., 1., 1.], [0., 0., 0., 0., 1., 1., 1., 1.]])
skills_needed_check = [[0], [1], [0, 1]]

# row = participants' responses to each question
# column = participants
responses_check.shape


def model(graded_responses, skills_needed: List[List[int]], prob_mistake=0.1, prob_guess=0.2):
    n_questions, n_participants = graded_responses.shape
    n_skills = max(map(max, skills_needed)) + 1
    
    with pyro.plate("participants_plate", n_participants):
        with pyro.plate("skills_plate", n_skills):
            skills = pyro.sample("skills", dist.Bernoulli(0.5), )
            
    for q in pyro.plate("questions_plate", n_questions):
        has_skills = reduce(operator.mul, [skills[i] for i in skills_needed[q]]).float()
        prob_correct = (has_skills * (1 - prob_mistake) + (1 - has_skills) * prob_guess)
        isCorrect = pyro.sample("isCorrect{}".format(q), dist.Bernoulli(prob_correct).to_event(1), obs=graded_responses[q])


def guide(graded_responses, skills_needed: List[List[int]], prob_mistake=0.1, prob_guess=0.2):
    _, n_participants = graded_responses.shape
    n_skills = max(map(max, skills_needed)) + 1
    
    skill_p = pyro.param("skill_p", 0.5 * torch.ones(n_skills,n_participants), constraint=constraints.unit_interval)
    
    with pyro.plate("participants_plate", n_participants):
        with pyro.plate("skills_plate", n_skills):
            skills = pyro.sample("skills", dist.Bernoulli(skill_p))
            
    return skills, skill_p


# smoke test for guide
guide(responses_check, skills_needed_check)

adam_params = {"lr": 0.05, "betas": (0.9, 0.999)}
optimizer = pyro.optim.Adam(adam_params)

# +
pyro.clear_param_store()
svi = pyro.infer.SVI(model=model, guide=guide, optim=optimizer, loss=pyro.infer.Trace_ELBO())

losses, skills_svi = [], []
for t in range(10_000):
    losses.append(svi.step(responses_check, skills_needed_check))
    skills_svi.append(pyro.param("skill_p"))
# -

d = pyro.get_param_store()

res_skill_0 = torch.vstack([s[0] for s in skills_svi])
res_skill_0 = res_skill_0.detach().numpy()
res_skill_0

# +
plt.subplot(1,2,1)
plt.semilogy(losses)

ax2 = plt.subplot(1,2,2)
for n in range(8):
    plt.plot(res_skill_0[:,n], label=str(n))
    
plt.legend()
# -

res_skill_1 = torch.vstack([s[1] for s in skills_svi])
res_skill_1 = res_skill_1.detach().numpy()
res_skill_1

# +
# same loss, skill_02
plt.subplot(1,2,1)
plt.semilogy(losses)

ax2 = plt.subplot(1,2,2)
for n in range(8):
    plt.plot(res_skill_1[:,n], label=str(n))
    
plt.legend()
# -

# ## Expected Answer

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
expected

expected["SVI skill_01 P(csharp)"] = res_skill_0[-1, :]
expected["SVI skill_02 P(sql)"] = res_skill_1[-1, :]

expected
