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
# * Trying to reproduce the functionality of the `sim()` function in the `arm` (**A**pplied **R**egression and **M**ultilevel modeling) package Ref [1]
#     - Original [source of `arm`](http://www.stat.columbia.edu/~gelman/arm/software/)
#     - Someone's github clone of `arm`, [link to the `sim()` function](https://github.com/suyusung/arm/blob/master/R/sim.R) function
# * Relevant discussions of `sim`
#     - [Accepted answer to the question Simulating draws/data from a fitted model with factor/character level inputs](https://stackoverflow.com/a/42151309/3587374)
#     - [Why does the sim function in Gelman's arm package simulate sigma from inverse chi square?](https://stats.stackexchange.com/questions/192996/why-does-the-sim-function-in-gelmans-arm-package-simulate-sigma-from-inverse-ch)
#     
# 1. A. Gelman and J. Hill, Data Analysis Using Regression and Multilevel/Hierarchical Models, Cambridge University Press (2007). 

# +
from typing import Optional, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2, t
from sklearn.metrics import mean_squared_error, r2_score


# -

# %matplotlib inline
# %reload_ext autoreload
# %autoreload 2
# %load_ext watermark

# %watermark -v -m -p numpy,matplotlib,pandas,sklearn,seaborn,statsmodels,scipy

# %watermark -gb

# +
def resid_std(resid: np.array, df: int):
    return np.sqrt(np.sum(np.square(resid)) / df)


def display(result, digits=1):
    model = result.model.__class__.__name__
    formula = result.model.formula
    print(f"{model}(formular = {formula})\n")
    table = pd.concat([result.params, result.bse], axis=1).rename(
        columns={0: "coef.est", 1: "coef.se"}
    )
    with pd.option_context("display.float_format", "{{:0.{}f}}".format(digits).format):
        print(table)

    print(f"n = {int(result.nobs)}, k = {len(result.params)}")
    print(
        f"residual sd = {resid_std(result.resid, result.nobs - len(result.params)):0.1f}, R-Squared = {result.rsquared:0.2f}"
    )


def sim(result, n, **kwargs):
    df = result.nobs - len(result.params)
    sigmas = resid_std(result.resid, df) * np.sqrt((df) / chi2.rvs(df, size=n))
    betas = np.random.multivariate_normal(
        result.params, result.cov_params(**kwargs), size=n
    )
    return betas, sigmas


def score(y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2_ = r2_score(y_true, y_pred)
    print(f"RMSE = {rmse:.2f}")
    print(f"R-Squared = {r2_:.2f}")


def _forward(X, weights, add_constant=False):
    return (
        np.dot(np.concatenate([np.ones((X.shape[0], 1)), X], axis=1), weights)
        if add_constant
        else np.dot(X, weights)
    )


def plot_fit(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    weights: np.array,
    axis_labels: Optional[Tuple[str, str]] = (None, None),
    error: Optional[Tuple[int, np.array]] = None,
    return_fig_ax=False,
    jitter: Optional[Union[float, np.array]] = 0,
    ax: Optional = None,
    add_constant=False,
):
    fig, ax = plt.subplots() if ax is None else (None, ax)
    if error is not None:
        for i in range(error[0]):
            ax.plot(
                data[x_col],
                _forward(
                    data[x_col].values[:, None],
                    error[1][i, :],
                    add_constant=add_constant,
                ),
                c="xkcd:grey",
                alpha=0.15,
            )
    ax.plot(
        data[x_col],
        _forward(data[x_col].values[:, None], weights, add_constant=add_constant),
        c="k",
    )
    ax.scatter(data[x_col] + jitter, data[y_col])
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])

    if return_fig_ax:
        return fig, ax


# +
font = {"size": 16}
mpl.rc("font", **font)

mpl.rcParams["figure.dpi"] = 100
# mpl.rcParams["font.sans-serif"] = "Arial"
# mpl.rcParams["font.family"] = "sans-serif"
# -

# ## `statsmodel` OLS
# * Taken from the `statsmodel`'s docs on [Ordinary Least Squares](https://www.statsmodels.org/dev/examples/notebooks/generated/ols.html)

np.random.default_rng(9876789)

nsample = 100
x = np.linspace(0, 10, 100)
X = np.column_stack((x, x ** 2))
beta = np.array([1, 0.1, 10])
e = np.random.normal(size=nsample)

sm_ols_example = pd.DataFrame(X, columns=["x1", "x2"])

sm_ols_example = sm.add_constant(sm_ols_example)
sm_ols_example["Y"] = np.dot(sm_ols_example, beta) + e

results = smf.ols("Y ~ x1 + x2", data=sm_ols_example).fit()
display(results, digits=4)

results.cov_params()

results.resid.std()

sigma_hat = resid_std(results.resid, results.nobs - len(results.params))
sigma_hat

beta = results.params
sigma = results.cov_params(scale=None)

B, s = sim(results, 1000, scale=None)

B.shape

# + tags=[]
sns.pairplot(pd.DataFrame(B))
# -

pd.DataFrame(B).mean(), pd.DataFrame(B).std()

pd.DataFrame(B).quantile([0.025, 0.975])

sigma_hat

pd.DataFrame(s).mean(), pd.DataFrame(s).std()

# ### Verifying sigma sampling
# * Not 100% sure which resid standard deviation, they both look similar
# * I think `e = np.random.normal(size=nsample)` default scale is `1.` so both look correct
# * Maybe I should try with a different scale

df_ = results.nobs - len(results.params)
sigmas_ = resid_std(results.resid, df_) * np.sqrt((df_) / chi2.rvs(df_, size=1000))
print(
    f"df = {df_} | mean sigma = {sigmas_.mean():0.2f} | std sigma = {sigmas_.std():0.2f}"
)

df_ = results.nobs - len(results.params)
sigmas_ = results.resid.std() * np.sqrt((df_) / chi2.rvs(df_, size=1000))
print(
    f"df = {df_} | mean sigma = {sigmas_.mean():0.2f} | std sigma = {sigmas_.std():0.2f}"
)

# ## 8.1 Fake-data simulation
# - From Chapter 8, Simulation for checking statistical procedures and model fits. Ref [1]

rng = np.random.default_rng(9876789)

x = np.arange(1, 6)
beta = np.array([1.4, 2.3])
sigma = 0.9
e = rng.normal(scale=sigma, size=len(x))

data = pd.DataFrame(x, columns=["x"])
data = sm.add_constant(data)
data["y"] = np.dot(data, beta) + e
data

results = smf.ols("y ~ x", data=data).fit()
display(results)

n_ = 1000
B, s = sim(results, n_, scale=None)

plot_fit(data, "x", "y", results.params.values, add_constant=True, error=(10, B))

B = pd.DataFrame(B)
s = pd.DataFrame(s)

print(f"a true = {beta[0]:9} | mean = {B.mean()[0]:0.1f} std = {B.std()[0]:0.1f}")
print(f"b true = {beta[1]:9} | mean = {B.mean()[1]:0.1f} std = {B.std()[1]:0.1f}")
print(f"sigma true = {sigma:5} | mean = {s.mean()[0]:0.1f} std = {s.std()[0]:0.1f}")

df_ = results.nobs - len(results.params)
sigmas_ = resid_std(results.resid, df_) * np.sqrt((df_) / chi2.rvs(df_, size=n_))
print(
    f"df = {df_} | mean sigma = {sigmas_.mean():0.2f} | std sigma = {sigmas_.std():0.2f}"
)

df_ = results.nobs - len(results.params)
sigmas_ = results.resid.std(ddof=int(df_)) * np.sqrt(df_ / chi2.rvs(df_, size=n_))
print(
    f"df = {df_} | mean sigma = {sigmas_.mean():0.1f} | std sigma = {sigmas_.std():0.1f}"
)

results.resid

np.sqrt(np.sum(np.square(results.resid)) / df_)

results.resid.std(ddof=int(df_)), resid_std(results.resid, df_)

cover_68 = np.abs(beta[1] - results.params[1]) < results.bse[1]
cover_95 = np.abs(beta[1] - results.params[1]) < 2 * results.bse[1]
print(f"68% coverage: {cover_68}\n95% coverage: {cover_95}")

# +
n = 1000
cover_68 = np.zeros(n)
cover_95 = np.zeros(n)
for i in range(n):
    e = np.random.normal(scale=sigma, size=len(x))
    data = pd.DataFrame(x, columns=["x"])
    data = sm.add_constant(data)
    data["y"] = np.dot(data, beta) + e
    results = smf.ols("y ~ x", data=data).fit()
    cover_68[i] = np.abs(beta[1] - results.params[1]) < results.bse[1]
    cover_95[i] = np.abs(beta[1] - results.params[1]) < 2 * results.bse[1]

print(f"Assuming Normal, which is wrong")
print(f"68% coverage: {cover_68.mean()}\n95% coverage: {cover_95.mean()}")

# +
n = 1000
cover_68 = np.zeros(n)
cover_95 = np.zeros(n)
t_68 = t.ppf(0.84, len(x) - 2)
t_95 = t.ppf(0.975, len(x) - 2)
for i in range(n):
    e = np.random.normal(scale=sigma, size=len(x))
    data = pd.DataFrame(x, columns=["x"])
    data = sm.add_constant(data)
    data["y"] = np.dot(data, beta) + e
    results = smf.ols("y ~ x", data=data).fit()
    cover_68[i] = np.abs(beta[1] - results.params[1]) < t_68 * results.bse[1]
    cover_95[i] = np.abs(beta[1] - results.params[1]) < t_95 * results.bse[1]

print(f"Assuming t-dist")
print(f"68% coverage: {cover_68.mean()}\n95% coverage: {cover_95.mean()}")

# +
n = 1000
cover_68 = np.zeros((n,2))
cover_95 = np.zeros((n,2))
t_68 = t.ppf(0.84, len(x) - 2)
t_95 = t.ppf(0.975, len(x) - 2)
for i in range(n):
    e = np.random.normal(scale=sigma, size=len(x))
    data = pd.DataFrame(x, columns=["x"])
    data = sm.add_constant(data)
    data["y"] = np.dot(data, beta) + e
    results = smf.ols("y ~ x", data=data).fit()
    cover_68[i,0] = np.abs(beta[0] - results.params[0]) < t_68 * results.bse[0]
    cover_68[i,1] = np.abs(beta[1] - results.params[1]) < t_68 * results.bse[1]
    cover_95[i,0] = np.abs(beta[0] - results.params[0]) < t_95 * results.bse[0]
    cover_95[i,1] = np.abs(beta[1] - results.params[1]) < t_95 * results.bse[1]

print(f"Assuming t-dist")
print(f"68% coverage: {cover_68.mean(0)} std: {cover_68.std(0)}\n95% coverage: {cover_95.mean(0)} std: {cover_95.std(0)}")
# -

# ## Larger Data

rng = np.random.default_rng(9876789)
nsample = 100
x = np.linspace(0, 10, 100)
X = np.column_stack((x, x ** 2))
sigma = 2
beta = np.array([1, 0.1, 10])
e = rng.normal(scale=sigma, size=nsample)
sm_ols_example = pd.DataFrame(X, columns=["x1", "x2"])
sm_ols_example = sm.add_constant(sm_ols_example)
sm_ols_example["Y"] = np.dot(sm_ols_example, beta) + e

results = smf.ols("Y ~ x1 + x2", data=sm_ols_example).fit()
display(results, digits=4)

n_ = 1000
B, s = sim(results, n_, scale=None)

B = pd.DataFrame(B)
s = pd.DataFrame(s)

print(f"b0 true = {beta[0]:9} | mean = {B.mean()[0]:0.2f} std = {B.std()[0]:0.2f}")
print(f"b1 true = {beta[1]:9} | mean = {B.mean()[1]:0.2f} std = {B.std()[1]:0.2f}")
print(f"b2 true = {beta[2]:9} | mean = {B.mean()[2]:0.2f} std = {B.std()[2]:0.2f}")
print(f"sigma true = {sigma:6} | mean = {s.mean()[0]:0.2f} std = {s.std()[0]:0.2f}")

# ### Verifying sigma sampling
# * This comparison seems to verify Gelman's approach

df_ = results.nobs - len(results.params)
sigmas_ = resid_std(results.resid, df_) * np.sqrt(df_ / chi2.rvs(df_, size=n_))
print(
    f"df = {df_} | mean sigma = {sigmas_.mean():0.2f} | std sigma = {sigmas_.std():0.2f}"
)

df_ = results.nobs - len(results.params)
sigmas_ = results.resid.std(ddof=int(df_)) * np.sqrt(df_ / chi2.rvs(df_, size=n_))
print(
    f"df = {df_} | mean sigma = {sigmas_.mean():0.1f} | std sigma = {sigmas_.std():0.1f}"
)


