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
# * [Naive Bayes model with PyMC3](https://discourse.pymc.io/t/naive-bayes-model-with-pymc3/2314)
# * [PyMC Tutorial #2: Estimating the Parameters of A Naïve Bayes (NB) Model ](http://alfan-farizki.blogspot.com/2015/07/pymc-tutorial-2-estimating-parameters.html)
# * [PyMC3 Models NaiveBayes](https://pymc3-models.readthedocs.io/en/latest/api/pymc3_models.models.html#module-pymc3_models.models.NaiveBayes)
#     - To learn more, you can read this section, watch a [video from PyData NYC 2017](https://www.youtube.com/watch?v=zGRnirbHWJ8), or check out the [slides](https://github.com/parsing-science/pydata_nyc_nov_2017).
#     - [Example Notebook](https://pymc3-models.readthedocs.io/en/latest/examples.html)
# * [(Bayesian) Naive Bayes algorithm](https://nbviewer.jupyter.org/github/parsing-science/pymc3_models/blob/master/notebooks/NaiveBayes.ipynb)
# * [Probabilistic Graphical Models](https://cs.brown.edu/courses/csci2950-p/lectures/2013-04-25_crfMaxProduct.pdf)
# * [Bayesian Naive Bayes, aka Dirichlet-Multinomial Classifiers](https://lingpipe-blog.com/2009/10/02/bayesian-naive-bayes-aka-dirichlet-multinomial-classifiers/)
# * [1.3. Naive Bayes](https://pymc-learn.readthedocs.io/en/latest/modules/naive_bayes.html#naive-bayes)
# * [ML in Python: Naive Bayes the hard way](https://healthyalgorithms.com/2013/04/13/ml-in-python-naive-bayes-the-hard-way/)
#     - [Notebook - really inefficient version of the Naive Bayes classifier](https://nbviewer.ipython.org/gist/anonymous/5380476)
# * [Bayesian Methods (Lab)](https://davidrosenberg.github.io/mlcourse/Archive/2015/Lectures/12.Lab.bayesian-methods.pdf)
#     - Source from [mlcourse by David Rosenberg](https://github.com/davidrosenberg/mlcourse)
# * [How to specify the prior probability for scikit-learn's Naive Bayes](https://stackoverflow.com/questions/30896367/how-to-specify-the-prior-probability-for-scikit-learns-naive-bayes)
# * [Naive Bayesian](https://saedsayad.com/naive_bayesian.htm)

# +
import numpy as np
import jax
import jax.numpy as jnp
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
# -

# %matplotlib inline
# %reload_ext autoreload
# %autoreload 2
# %load_ext watermark

# %watermark -v -m -p numpy,jax,pandas,matplotlib,arviz,numpyro

# %watermark -gb

key = jax.random.PRNGKey(0)
key

keys = jax.random.split(key, 2)
d_X = dist.Bernoulli(jnp.array((0.91, 0.02, 0.90, 0.32, 0.66, 0.71, 0.32, 0.70, 0.94, 0.49)))
d_y = dist.Bernoulli(jnp.array((0.8)))
X = d_X.sample(keys[0], (300,))
y = d_y.sample(keys[1], (300,))

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

X_train

X_train.shape, y_train.shape


def model(X, num_classes, y=None,):
    num_items, num_features = X.shape
    pi = numpyro.sample("pi", dist.Dirichlet(jnp.ones(num_classes)))
    plate_D = numpyro.plate("D", num_features, dim=None)
    
    with plate_D:
        with numpyro.plate("C", num_classes):
            theta = numpyro.sample("theta_jc", dist.Beta(1,1))
            
    with numpyro.plate("N", num_items, dim=-2):
        y = numpyro.sample("Y_i", dist.Categorical(pi), obs=y)
        with plate_D:
            x = numpyro.sample("X_ij", dist.Bernoulli(theta[y]), obs=X)


numpyro.render_model(model, (X_train[:10,:], 2, y_train[:10]), render_distributions=True)

nuts_kernel = NUTS(model)
mcmc = MCMC(nuts_kernel, num_warmup=500, num_samples=1000, num_chains=4)
mcmc.run(key, X_train, 2, y_train, extra_fields=("potential_energy",))
mcmc.print_summary()

ds = az.from_numpyro(mcmc)

az.plot_trace(ds);

from numpyro.infer import Predictive

predictive = Predictive(model, mcmc.get_samples())

predictions = predictive(jax.random.PRNGKey(3), X_test, 2)['Y_i']

X_test.shape

predictions.shape

predictions.squeeze().mean()

clf = BernoulliNB(alpha=1)

clf.fit(X_train, y_train)

clf.class_count_ / clf.class_count_.sum()

clf.feature_count_.T

np.exp(clf.feature_log_prob_).T

clf.score(X_test, y_test)

# ## MLE

from numpyro.infer import SVI, Trace_ELBO, autoguide


def model(X, num_classes, y=None,):
    num_items, num_features = X.shape
    pi = numpyro.sample("pi", dist.Dirichlet(jnp.ones(num_classes)))
    plate_D = numpyro.plate("D", num_features, dim=None)
    
    with plate_D:
        with numpyro.plate("C", num_classes):
            theta = numpyro.param("theta_jc", jnp.ones((num_classes,num_features)) * 0.5, constraint=dist.constraints.unit_interval)
            
    with numpyro.plate("N", num_items, dim=-2):
        y = numpyro.sample("Y_i", dist.Categorical(pi), obs=y)
        with plate_D:
            x = numpyro.sample("X_ij", dist.Bernoulli(theta[y]), obs=X)


def guide(X, num_classes, y=None,):
    class_prior = numpyro.param("class_prior", jnp.ones(num_classes), constraint=dist.constraints.positive)
    numpyro.sample("pi", dist.Dirichlet(class_prior))


guide = autoguide.AutoDiagonalNormal(model)   

optimizer = numpyro.optim.Adam(0.0005)
svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
svi_result = svi.run(jax.random.PRNGKey(15), 2000, X_train, 2, y_train)
params = svi_result.params

params


