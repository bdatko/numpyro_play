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
# * Reproducing the [Figure 2.29 from MBML](https://www.mbmlbook.com/LearningSkills_Learning_the_guess_probabilities.html), shown below
# * Iterations and experiments were driven by the [discussion found on the Pyro forum](https://forum.pyro.ai/t/log-probability-of-model/3241?u=bdatko)
#
# ![image.png](attachment:cb44a245-9c7a-44b0-8b11-244dbc02b652.png)
# ![image.png](attachment:797c8ac1-ad9c-4de5-ae0c-38890a5083b4.png)
# >**Figure 2.29**: (left) Overall negative log probability for the original model and the model with learned guess probabilities. The lower red bar indicates that learning the guess probabilities gives a substantially better model, according to this metric. (right) Negative log probability for each skill, showing that the improvement varies from skill to skill. 
#
# * The figure is **not** the `log_density` of the model it is the negative log probability of the ground truth. For a participant with $skill_i$ the negative log probability is . 
#
# $$-log(p(skill_i = truth_i))$$
#
# where $truth_i$ is an indicator variable of having $skill_i$ and the probability of each skill is $p(skill_i)$ ~ $Bernoulli(\theta_i)$
#
# Further details from the text:
#
# > A common metric to use is the **probability of the ground truth** values under the inferred distributions. Sometimes it is convenient to take the logarithm of the probability, since this gives a more manageable number when the probability is very small. When we use the logarithm of the probability, the metric is referred to as the log probability. So, if the inferred probability of a person having a particular skill is $p$, then the log probability is $log(p)$ if the person has the skill and $log(1−p)$ if they don’t. If the person does have the skill then the best possible prediction is $p=1.0$, which gives log probability of $log(1.0)=0$ (the logarithm of one is zero). A less confident prediction, such as $p=0.8$ will give a log probability with a negative value, in this case $log(0.8)=−0.097$. The worst possible prediction of $p=0.0$ gives a log probability of negative infinity. ...

# +
import operator
from functools import reduce
from typing import Dict, List

import arviz as az
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
from jax.scipy.special import logsumexp

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, DiscreteHMCGibbs, log_likelihood
from numpyro.infer.util import log_density, potential_energy


# -

# %matplotlib inline
# %reload_ext autoreload
# %autoreload 2
# %load_ext watermark

# %watermark -v -m -p arviz,jax,matplotlib,numpy,pandas,scipy,numpyro

# %watermark -gb

def neg_log_proba_score(posterior_samples: Dict, params_sites: List[str], y_true):
    """
    Calculates the the negative log probability of the ground truth, the self assessed skills.
    :param posterior_samples Dict: dictionary of samples from the posterior.
    :param params_sites List[str]: a list of params to compute proba
    :y_true array-like dtype == int: array of indicator variables for skill of participants
    """
    assert np.issubdtype(y_true.dtype, np.integer)
    proba = np.zeros((len(params_sites), posterior_samples[params_sites[0]].shape[-1]))
    assert proba.shape == y_true.shape
    for i, param in enumerate(params_sites):
        proba[i, :] = np.mean(posterior_samples[param], axis=0)

    score = scipy.stats.bernoulli(proba).pmf(y_true)
    score[score == 0.0] = np.finfo(float).eps

    return -np.log(score)


rng_key = jax.random.PRNGKey(2)

# ### Get Data

# +
raw_data = pd.read_csv(
    "http://www.mbmlbook.com/Downloads/LearningSkills_Real_Data_Experiments-Original-Inputs-RawResponsesAsDictionary.csv"
)
self_assessed = raw_data.iloc[1:, 1:8].copy()
self_assessed = self_assessed.astype(int)

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

# ### Define models and run inference

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

mcmc_00 = MCMC(
    kernel, num_warmup=200, num_samples=1000, num_chains=4, jit_model_args=False
)
mcmc_00.run(
    rng_key,
    jnp.array(responses),
    skills_needed,
    extra_fields=(
        "z",
        "hmc_state.potential_energy",
        "hmc_state.z",
        "rng_key",
        "hmc_state.rng_key",
    ),
)
mcmc_00.print_summary()
# -

ds = az.from_numpyro(mcmc_00)

# + tags=[]
az.plot_trace(ds);

# + tags=[]
log_density_model_00, model_00_trace = log_density(
    model_00,
    (jnp.array(responses), skills_needed),
    dict(prob_mistake=0.1, prob_guess=0.2),
    {key: value.mean(0) for key, value in mcmc_00.get_samples().items()},
)
log_density_model_00


# -

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
        prob_correct = (
            has_skills * (1 - prob_mistake) + (1 - has_skills) * prob_guess[q]
        )
        isCorrect = numpyro.sample(
            "isCorrect_{}".format(q),
            dist.Bernoulli(prob_correct).to_event(1),
            obs=graded_responses[q],
        )


# + tags=[]
nuts_kernel = NUTS(model_02)

kernel = DiscreteHMCGibbs(nuts_kernel, modified=True)

mcmc_02 = MCMC(kernel, num_warmup=200, num_samples=1000, num_chains=4)
mcmc_02.run(
    rng_key,
    jnp.array(responses),
    skills_needed,
    extra_fields=(
        "z",
        "hmc_state.potential_energy",
        "hmc_state.z",
        "rng_key",
        "hmc_state.rng_key",
    ),
)
mcmc_02.print_summary()
# -

ds = az.from_numpyro(mcmc_02)

az.plot_trace(ds);

# ### Model Comparison

# + tags=[]
log_density_model_02, model_02_trace = log_density(
    model_02,
    (jnp.array(responses), skills_needed),
    dict(prob_mistake=0.1),
    {key: value.mean(0) for key, value in mcmc_02.get_samples().items()},
)
log_density_model_02

# +
pe_model_00 = mcmc_00.get_extra_fields()["hmc_state.potential_energy"]
pe_model_02 = mcmc_02.get_extra_fields()["hmc_state.potential_energy"]

print(
    "Expected log joint density of model_00: {:.2f} +/- {:.2f}".format(
        np.mean(-pe_model_00), np.std(-pe_model_00)
    )
)
print(
    "Expected log joint density of model_02: {:.2f} +/- {:.2f}".format(
        np.mean(-pe_model_02), np.std(-pe_model_02)
    )
)
# -

pd.DataFrame(
    np.array([np.mean(pe_model_00), np.mean(pe_model_02)])[None, :],
    columns=["Original", "Learned"],
    index=["Overall"],
).plot(
    kind="bar",
    color=["b", "r"],
    ylabel="negative Expected log density",
    yerr=[[np.std(pe_model_00)], [np.std(pe_model_02)]],
)

print(
    "Expected log joint density of model_00 from `log_density`: {:.2f}".format(
        log_density_model_00
    )
)
print(
    "Expected log joint density of model_02 from `log_density`: {:.2f}".format(
        log_density_model_02
    )
)

pd.DataFrame(
    -np.array([log_density_model_00, log_density_model_02])[None, :],
    columns=["Original", "Learned"],
    index=["Overall"],
).plot(kind="bar", color=["b", "r"], ylabel="negative log density")

# #### **log_likelihood** 
#
# signature: `log_likelihood(model, posterior_samples, *args, parallel=False, batch_ndims=1, **kwargs)`
#
# * A good example of `log_likelihood` is from the [Example: Baseball Batting Average](http://num.pyro.ai/en/stable/examples/baseball.html#example-baseball-batting-average)
# * The code below calculates the Log pointwise predictive density

post_loglik_00 = log_likelihood(model_00, mcmc_00.get_samples(), jnp.array(responses), skills_needed,)
post_loglik_00_res = np.concatenate([obs[:,None] for obs in post_loglik_00.values()],axis=1)
exp_log_density_00 = logsumexp(post_loglik_00_res, axis=0) - jnp.log(jnp.shape(post_loglik_00_res)[0])

post_loglik_02 = log_likelihood(model_02, mcmc_02.get_samples(), jnp.array(responses), skills_needed,)
post_loglik_02_res = np.concatenate([obs[:,None] for obs in post_loglik_02.values()],axis=1)
exp_log_density_02 = logsumexp(post_loglik_02_res, axis=0) - jnp.log(jnp.shape(post_loglik_02_res)[0])

pd.DataFrame(
    np.array([np.sum(exp_log_density_00), np.sum(exp_log_density_02)])[None, :],
    columns=["Original", "Learned"],
    index=["Overall"],
).plot(
    kind="bar",
    color=["b", "r"],
    ylabel="Log pointwise predictive density",
)

# #### Negative log probability of the ground truth

# +
neg_log_proba_model_00 = neg_log_proba_score(
    mcmc_00.get_samples(),
    ["skill_" + str(i) for i in range(7)],
    self_assessed.values.T,
)

neg_log_proba_model_02 = neg_log_proba_score(
    mcmc_02.get_samples(),
    ["skill_" + str(i) for i in range(7)],
    self_assessed.values.T,
)
# -

# #### Figure 2.29(a)

pd.DataFrame(
    np.array([neg_log_proba_model_00.mean(), neg_log_proba_model_02.mean()])[None, :],
    columns=["Original", "Learned"],
    index=["Overall"],
).plot(kind="bar", color=["b", "r"])

# #### Figure 2.29(b)

pd.DataFrame(
    np.concatenate(
        [
            neg_log_proba_model_00.mean(1)[:, None],
            neg_log_proba_model_02.mean(1)[:, None],
        ],
        axis=1,
    ),
    columns=["Original", "Learned"],
    index=["Core", "OOP", "Life Cycle", "Web Apps Skills", "Desktop apps", "SQL", "C#"],
).plot(kind="bar", color=["b", "r"])


# ### Odds and Ends
# * failed attempts to reproduce the Figure 2.29
# * read at your own peril

def get_proba(
    posterior_samples: Dict, params_sites: List[str], negative_log_proba=False
):
    """
    :param posterior_samples Dict: dictionary of samples from the posterior.
    :param params_sites List[str]: a list of params to compute proba
    :param negative_log_proba bool: flag to return either probability or negative log probability
    """
    proba = np.zeros((len(params_sites), posterior_samples[params_sites[0]].shape[-1]))
    for i, param in enumerate(params_sites):
        proba[i, :] = np.mean(posterior_samples[param], axis=0)

    if negative_log_proba:
        proba[proba == 0.0] = np.finfo(float).eps
        proba = -np.log(proba)

    return proba


self_assessed.astype(int).values.dtype

# + tags=[]
self_assessed.astype(int).values.T.shape
# -

proba = get_proba(
    mcmc_00.get_samples(),
    ["skill_" + str(i) for i in range(7)],
    negative_log_proba=False,
).T

proba.shape



# +
proba_00 = get_proba(
    mcmc_00.get_samples(),
    ["skill_" + str(i) for i in range(7)],
    negative_log_proba=False,
).T

rv_00 = scipy.stats.bernoulli(proba_00)

proba_model_00 = rv_00.pmf(self_assessed.astype(int).values)

proba_model_00[proba_model_00 == 0.0] = np.finfo(float).eps

neg_log_proba_model_00 = -np.log(proba_model_00)

(-np.log(proba_model_00)).mean()

# +
proba_02 = get_proba(
    mcmc_02.get_samples(),
    ["skill_" + str(i) for i in range(7)],
    negative_log_proba=False,
).T

rv_02 = scipy.stats.bernoulli(proba_02)

proba_model_02 = rv_02.pmf(self_assessed.astype(int).values)

proba_model_02[proba_model_02 == 0.0] = np.finfo(float).eps

neg_log_proba_model_02 = -np.log(proba_model_02)

(-np.log(proba_model_02)).mean()
# -

pd.DataFrame(
    np.array([neg_log_proba_model_00.mean(), neg_log_proba_model_02.mean()])[None, :],
    columns=["Original", "Learned"],
    index=["Overall"],
).plot(kind="bar", color=["b", "r"])

pd.DataFrame(
    np.concatenate(
        [
            neg_log_proba_model_00.mean(0)[:, None],
            neg_log_proba_model_02.mean(0)[:, None],
        ],
        axis=1,
    ),
    columns=["Original", "Learned"],
    index=["Core", "OOP", "Life Cycle", "Web Apps Skills", "Desktop apps", "SQL", "C#"],
).plot(kind="bar", color=["b", "r"])

-dist.Bernoulli(proba_02).log_prob(self_assessed.astype(int).values)

neg_log_proba_model_00 = get_proba(
    mcmc_00.get_samples(),
    ["skill_" + str(i) for i in range(7)],
    negative_log_proba=True,
).mean(1)

neg_log_proba_model_02 = get_proba(
    mcmc_02.get_samples(),
    ["skill_" + str(i) for i in range(7)],
    negative_log_proba=True,
).mean(1)

pd.DataFrame(
    np.array([neg_log_proba_model_00.mean(), neg_log_proba_model_02.mean()])[None, :],
    columns=["Original", "Learned"],
    index=["Overall"],
).plot(
    kind="bar",
    color=["b", "r"],
    yerr=[[neg_log_proba_model_00.std()], [neg_log_proba_model_02.std()]],
)

# +
neg_log_proba_model_00_std = get_proba(
    mcmc_00.get_samples(),
    ["skill_" + str(i) for i in range(7)],
    negative_log_proba=True,
).std(1)

neg_log_proba_model_02_std = get_proba(
    mcmc_02.get_samples(),
    ["skill_" + str(i) for i in range(7)],
    negative_log_proba=True,
).std(1)
# -

std = np.concatenate(
    [neg_log_proba_model_00_std[:, None], neg_log_proba_model_02_std[:, None]], axis=1
).T

pd.DataFrame(
    np.concatenate(
        [neg_log_proba_model_00[:, None], neg_log_proba_model_02[:, None]], axis=1
    ),
    columns=["Original", "Learned"],
    index=["Core", "OOP", "Life Cycle", "Web Apps Skills", "Desktop apps", "SQL", "C#"],
).plot(kind="bar", color=["b", "r"])

pd.DataFrame(
    np.concatenate(
        [neg_log_proba_model_00[:, None], neg_log_proba_model_02[:, None]], axis=1
    ),
    columns=["Original", "Learned"],
    index=["Core", "OOP", "Life Cycle", "Web Apps Skills", "Desktop apps", "SQL", "C#"],
).plot(kind="bar", color=["b", "r"], yerr=std)

[
    -np.log(mcmc_00.get_samples()[s].mean(0)).mean()
    for s in ["skill_" + str(i) for i in range(7)]
]

# +
res_00 = np.zeros((7, 22))
for i in range(7):
    s = "skill_" + str(i)
    res_00[i, :] = np.mean(mcmc_00.get_samples()[s], axis=0)

res_00[res_00 == 0.0] = np.finfo(float).eps

neg_log_proba_model_00 = (-np.log(res_00)).mean(1)

# +
res_02 = np.zeros((7, 22))
for i in range(7):
    s = "skill_" + str(i)
    res_02[i, :] = np.mean(mcmc_02.get_samples()[s], axis=0)

res_02[res_02 == 0.0] = np.finfo(float).eps

neg_log_proba_model_02 = (-np.log(res_02)).mean(1)
# -

np.array([(-np.log(res_00)).mean(), (-np.log(res_02)).mean()])[None, :]

np.finfo(float).eps

pd.DataFrame(
    np.array([(-np.log(res_00)).mean(), (-np.log(res_02)).mean()])[None, :],
    columns=["Original", "Learned"],
    index=["Overall"],
).plot(kind="bar", color=["b", "r"])

pd.DataFrame(
    np.concatenate(
        [neg_log_proba_model_00[:, None], neg_log_proba_model_02[:, None]], axis=1
    ),
    columns=["Original", "Learned"],
    index=["Core", "OOP", "Life Cycle", "Web Apps Skills", "Desktop apps", "SQL", "C#"],
).plot(kind="bar", color=["b", "r"])

[
    -np.log(mcmc_02.get_samples()[s].mean(0)).mean()
    for s in ["skill_" + str(i) for i in range(7)]
]

neg_log_proba_model_00 = np.array(
    [
        -np.log(mcmc_00.get_samples()[s].mean(0).mean())
        for s in ["skill_" + str(i) for i in range(7)]
    ]
)

neg_log_proba_model_02 = np.array(
    [
        -np.log(mcmc_02.get_samples()[s].mean(0).mean())
        for s in ["skill_" + str(i) for i in range(7)]
    ]
)

# +
neg_log_proba_model_00 = np.array(
    [
        -np.log(mcmc_00.get_samples()[s].mean(0).mean())
        for s in ["skill_" + str(i) for i in range(7)]
    ]
)

neg_log_proba_model_02 = np.array(
    [
        -np.log(mcmc_02.get_samples()[s].mean(0).mean())
        for s in ["skill_" + str(i) for i in range(7)]
    ]
)


pd.DataFrame(
    np.concatenate(
        [neg_log_proba_model_00[:, None], neg_log_proba_model_02[:, None]], axis=1
    ),
    columns=["Original", "Learned"],
    index=["Core", "OOP", "Life Cycle", "Web Apps", "Desktop apps", "SQL", "C#"],
).plot(kind="bar", color=["b", "r"])
# -


