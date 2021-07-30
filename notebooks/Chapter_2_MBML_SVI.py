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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyro
import pyro.distributions as dist
import torch
from pyro import poutine
from pyro.infer import (
    SVI,
    Trace_ELBO,
    TraceGraph_ELBO,
    config_enumerate,
    infer_discrete,
)
from pyro.ops.indexing import Vindex
from torch.distributions import constraints
from tqdm import tqdm

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

responses_check = torch.tensor(
    [
        [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
    ]
)
skills_needed_check = [[0], [1], [0, 1]]

# row = participants' responses to each question
# column = participants
responses_check.shape


def model(
    graded_responses, skills_needed: List[List[int]], prob_mistake=0.1, prob_guess=0.2
):
    n_questions, n_participants = graded_responses.shape
    n_skills = max(map(max, skills_needed)) + 1

    with pyro.plate("participants_plate", n_participants):
        with pyro.plate("skills_plate", n_skills):
            skills = pyro.sample("skills", dist.Bernoulli(0.5),)

    for q in pyro.plate("questions_plate", n_questions):
        has_skills = reduce(operator.mul, [skills[i] for i in skills_needed[q]]).float()
        prob_correct = has_skills * (1 - prob_mistake) + (1 - has_skills) * prob_guess
        isCorrect = pyro.sample(
            "isCorrect{}".format(q),
            dist.Bernoulli(prob_correct).to_event(1),
            obs=graded_responses[q],
        )


def guide(
    graded_responses, skills_needed: List[List[int]], prob_mistake=0.1, prob_guess=0.2
):
    _, n_participants = graded_responses.shape
    n_skills = max(map(max, skills_needed)) + 1

    skill_p = pyro.param(
        "skill_p",
        0.5 * torch.ones(n_skills, n_participants),
        constraint=constraints.unit_interval,
    )

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
svi = pyro.infer.SVI(model=model, guide=guide, optim=optimizer, loss=Trace_ELBO())

losses, skills_svi = [], []
for _ in tqdm(range(10_000)):
    losses.append(svi.step(responses_check, skills_needed_check))
    skills_svi.append(pyro.param("skill_p"))
# -

d = pyro.get_param_store()

res_skill_0 = torch.vstack([s[0] for s in skills_svi])
res_skill_0 = res_skill_0.detach().numpy()
res_skill_0

# +
plt.subplot(1, 2, 1)
plt.semilogy(losses)

ax2 = plt.subplot(1, 2, 2)
for n in range(8):
    plt.plot(res_skill_0[:, n], label=str(n))

plt.legend()
# -

res_skill_1 = torch.vstack([s[1] for s in skills_svi])
res_skill_1 = res_skill_1.detach().numpy()
res_skill_1

# +
# same loss, skill_02
plt.subplot(1, 2, 1)
plt.semilogy(losses)

ax2 = plt.subplot(1, 2, 2)
for n in range(8):
    plt.plot(res_skill_1[:, n], label=str(n))

plt.legend()
# -

expected["Trace_ELBO P(csharp)"] = res_skill_0[-1, :]
expected["Trace_ELBO P(sql)"] = res_skill_1[-1, :]

expected

# ## Change loss to `TraceGraph_ELBO`

# +
pyro.clear_param_store()
svi = pyro.infer.SVI(model=model, guide=guide, optim=optimizer, loss=TraceGraph_ELBO())

losses, skills_svi = [], []
for _ in tqdm(range(10_000)):
    losses.append(svi.step(responses_check, skills_needed_check))
    skills_svi.append(pyro.param("skill_p"))
# -

d = pyro.get_param_store()

res_skill_0 = torch.vstack([s[0] for s in skills_svi])
res_skill_0 = res_skill_0.detach().numpy()
res_skill_0

# +
plt.subplot(1, 2, 1)
plt.semilogy(losses)

ax2 = plt.subplot(1, 2, 2)
for n in range(8):
    plt.plot(res_skill_0[:, n], label=str(n))

plt.legend()
# -

res_skill_1 = torch.vstack([s[1] for s in skills_svi])
res_skill_1 = res_skill_1.detach().numpy()
res_skill_1

# +
# same loss, skill_02
plt.subplot(1, 2, 1)
plt.semilogy(losses)

ax2 = plt.subplot(1, 2, 2)
for n in range(8):
    plt.plot(res_skill_1[:, n], label=str(n))

plt.legend()
# -

expected["TraceGraph_ELBO P(csharp)"] = res_skill_0[-1, :]
expected["TraceGraph_ELBO P(sql)"] = res_skill_1[-1, :]

expected


# #### Reducing Variance with Data-Dependent Baselines

def guide_base(
    graded_responses, skills_needed: List[List[int]], prob_mistake=0.1, prob_guess=0.2
):
    _, n_participants = graded_responses.shape
    n_skills = max(map(max, skills_needed)) + 1

    skill_p = pyro.param(
        "skill_p",
        0.5 * torch.ones(n_skills, n_participants),
        constraint=constraints.unit_interval,
    )

    with pyro.plate("participants_plate", n_participants):
        with pyro.plate("skills_plate", n_skills):
            skills = pyro.sample(
                "skills",
                dist.Bernoulli(skill_p),
                infer=dict(
                    baseline={"use_decaying_avg_baseline": True, "baseline_beta": 0.9}
                ),
            )

    return skills, skill_p


# +
pyro.clear_param_store()
svi = pyro.infer.SVI(
    model=model, guide=guide_base, optim=optimizer, loss=TraceGraph_ELBO()
)

losses, skills_svi = [], []
for _ in tqdm(range(10_000)):
    losses.append(svi.step(responses_check, skills_needed_check))
    skills_svi.append(pyro.param("skill_p"))
# -

d = pyro.get_param_store()

# +
res_skill_0 = torch.vstack([s[0] for s in skills_svi])
res_skill_0 = res_skill_0.detach().numpy()
res_skill_0

res_skill_1 = torch.vstack([s[1] for s in skills_svi])
res_skill_1 = res_skill_1.detach().numpy()
res_skill_1

expected["decay TraceGraph_ELBO P(csharp)"] = res_skill_0[-1, :]
expected["decay TraceGraph_ELBO P(sql)"] = res_skill_1[-1, :]

# +
plt.subplot(1, 2, 1)
plt.semilogy(losses)

ax2 = plt.subplot(1, 2, 2)
for n in range(8):
    plt.plot(res_skill_0[:, n], label=str(n))

plt.legend()
# -

expected

# ## Decay experiments

from functools import partial
from sklearn.metrics import mean_squared_error, median_absolute_error, r2_score

rmse = partial(mean_squared_error, squared=False)

metrics = {
    "mse": mean_squared_error,
    "mae": median_absolute_error,
    "r2_score": r2_score,
    "rmse": rmse,
}


# +
def model_decay(
    graded_responses,
    skills_needed: List[List[int]],
    prob_mistake=0.1,
    prob_guess=0.2,
    baseline_beta=0.9,
):
    n_questions, n_participants = graded_responses.shape
    n_skills = max(map(max, skills_needed)) + 1

    with pyro.plate("participants_plate", n_participants):
        with pyro.plate("skills_plate", n_skills):
            skills = pyro.sample("skills", dist.Bernoulli(0.5),)

    for q in pyro.plate("questions_plate", n_questions):
        has_skills = reduce(operator.mul, [skills[i] for i in skills_needed[q]]).float()
        prob_correct = has_skills * (1 - prob_mistake) + (1 - has_skills) * prob_guess
        isCorrect = pyro.sample(
            "isCorrect{}".format(q),
            dist.Bernoulli(prob_correct).to_event(1),
            obs=graded_responses[q],
        )


def guide_decay(
    graded_responses,
    skills_needed: List[List[int]],
    prob_mistake=0.1,
    prob_guess=0.2,
    baseline_beta=0.9,
):
    _, n_participants = graded_responses.shape
    n_skills = max(map(max, skills_needed)) + 1

    skill_p = pyro.param(
        "skill_p",
        0.5 * torch.ones(n_skills, n_participants),
        constraint=constraints.unit_interval,
    )

    with pyro.plate("participants_plate", n_participants):
        with pyro.plate("skills_plate", n_skills):
            skills = pyro.sample(
                "skills",
                dist.Bernoulli(skill_p),
                infer=dict(
                    baseline={
                        "use_decaying_avg_baseline": True,
                        "baseline_beta": baseline_beta,
                    }
                ),
            )

    return skills, skill_p


# -

columns = [
    "P(csharp) mse",
    "P(sql) mse",
    "P(csharp) mae",
    "P(sql) mae",
    "P(csharp) r2_score",
    "P(sql) r2_score",
    "P(csharp) rmse",
    "P(sql) rmse",
]

result = []
index = np.linspace(-1., 1., 20)
for baseline in index:
    pyro.clear_param_store()
    guide = partial(guide_decay, baseline_beta=baseline)
    svi = pyro.infer.SVI(
        model=model_decay, guide=guide, optim=optimizer, loss=TraceGraph_ELBO()
    )

    losses, skills_svi = [], []
    for _ in tqdm(range(5000)):
        losses.append(svi.step(responses_check, skills_needed_check))
        skills_svi.append(pyro.param("skill_p"))

    res_skill_0 = torch.vstack([s[0] for s in skills_svi])
    res_skill_0 = res_skill_0.detach().numpy()

    res_skill_1 = torch.vstack([s[1] for s in skills_svi])
    res_skill_1 = res_skill_1.detach().numpy()

    row = []
    for metric in metrics.items():
        name, score = metric
        res_00 = score(expected["P(csharp)"], res_skill_0[-1, :])
        res_01 = score(expected["P(sql)"], res_skill_1[-1, :])
        row.extend([res_00, res_01])

    result.append(tuple(row))

expected

default = {}
for metric in metrics.items():
    name, score = metric
    default_res_00 = score(expected["P(csharp)"], expected["Trace_ELBO P(csharp)"])
    default_res_01 = score(expected["P(sql)"], expected["Trace_ELBO P(sql)"])
    default_res_02 = score(expected["P(csharp)"], expected["TraceGraph_ELBO P(csharp)"])
    default_res_03 = score(expected["P(sql)"], expected["TraceGraph_ELBO P(sql)"])
    default[name + "P(csharp)" + "Trace_ELBO"] = default_res_00
    default[name + "P(sql)" + "Trace_ELBO"] = default_res_01
    default[name + "P(csharp)" + "TraceGraph_ELBO"] = default_res_02
    default[name + "P(sql)" + "TraceGraph_ELBO"] = default_res_03
    print(
        f"{name:10} Trace_ELBO | P(csharp) = {default_res_00:0.4f} P(sql) = {default_res_01:0.4f} | TraceGraph_ELBO | P(csharp) = {default_res_02:0.4f} P(sql) = {default_res_03:0.4f}"
    )

result_df = pd.DataFrame(result, columns=columns, index=index)

result_df

result_df[list(set(columns) - set(["P(csharp) r2_score", "P(sql) r2_score"]))].iloc[1:].plot(
    marker="o"
)

result_df[["P(csharp) rmse", "P(sql) rmse"]].iloc[1:].plot(marker="o")
plt.hlines(default["rmseP(csharp)Trace_ELBO"], -1., 1., color="r", linestyles="--", label="baseline P(csharp)")
plt.hlines(default["rmseP(sql)Trace_ELBO"], -1., 1., color="k", linestyles="--", label="baseline P(sql)")
plt.legend()

result_df[["P(csharp) mse", "P(sql) mse"]].iloc[1:].plot(marker="o")
plt.hlines(default["mseP(csharp)Trace_ELBO"], -1., 1., color="r", linestyles="--", label="baseline P(csharp)")
plt.hlines(default["mseP(sql)Trace_ELBO"], -1., 1., color="k", linestyles="--", label="baseline P(sql)")
plt.legend()

result_df[["P(csharp) r2_score", "P(sql) r2_score"]].iloc[1:].plot(marker="o")
plt.hlines(default["r2_scoreP(csharp)Trace_ELBO"], -1., 1., color="r", linestyles="--", label="baseline P(csharp)")
plt.hlines(default["r2_scoreP(sql)Trace_ELBO"], -1., 1., color="k", linestyles="--", label="baseline P(sql)")
plt.legend()


# ### Using the best param

def guide_base(
    graded_responses, skills_needed: List[List[int]], prob_mistake=0.1, prob_guess=0.2
):
    _, n_participants = graded_responses.shape
    n_skills = max(map(max, skills_needed)) + 1

    skill_p = pyro.param(
        "skill_p",
        0.5 * torch.ones(n_skills, n_participants),
        constraint=constraints.unit_interval,
    )

    with pyro.plate("participants_plate", n_participants):
        with pyro.plate("skills_plate", n_skills):
            skills = pyro.sample(
                "skills",
                dist.Bernoulli(skill_p),
                infer=dict(
                    baseline={"use_decaying_avg_baseline": True, "baseline_beta": 0.368421}
                ),
            )

    return skills, skill_p


# +
pyro.clear_param_store()
svi = pyro.infer.SVI(
    model=model, guide=guide_base, optim=optimizer, loss=TraceGraph_ELBO()
)

losses, skills_svi = [], []
for _ in tqdm(range(10_000)):
    losses.append(svi.step(responses_check, skills_needed_check))
    skills_svi.append(pyro.param("skill_p"))

# +
res_skill_0 = torch.vstack([s[0] for s in skills_svi])
res_skill_0 = res_skill_0.detach().numpy()
res_skill_0

res_skill_1 = torch.vstack([s[1] for s in skills_svi])
res_skill_1 = res_skill_1.detach().numpy()
res_skill_1

expected["best decay TraceGraph_ELBO P(csharp)"] = res_skill_0[-1, :]
expected["best decay TraceGraph_ELBO P(sql)"] = res_skill_1[-1, :]
# -

expected


