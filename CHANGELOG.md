## v0.3.0 (2021-08-29)

### Feat

- **notebooks/Gelman_arm_sim_statsmodels.***: add excerpt from Gelman to add context
- **notebooks/Section\-3.6\-of\-Chapter\-3\--\-Inferences\-with\-binomials.***: add watermark to notebook notebooks/Section\ 3.6\ of\ Chapter\ 3\ -\ Inferences\ with\ binomials.*
- **notebooks/pyro_importance_sampling.***: add watermark to notebook notebooks/pyro_importance_sampling
- **notebooks/numpyro_params.***: add params explore notebook
- **notebooks/discrete_latent.***: add watermark notebook to notebooks/discrete_latent
- **notebooks/Config_Enumerate_in_Pyro.***: add watermark to notebook notebooks/Config_Enumerate_in_Pyro
- **notebooks/Chapter_2_MBML_SVI.***: add watermark to notebook notebooks/Chapter_2_MBML_SVI
- **notebooks/Chapter_2_MBML_numpyro_experiments.***: add watermark to notebook Chapter_2_MBML_numpyro_experiments
- **notebooks/Chapter_2_MBML_log_probability.***: add watermark to Chapter_2_MBML_log_probability
- **notebooks/Bayesian_Naive_Bayes.***: add watermark to Bayesian Naive Bayes notebook
- **notebooks/baseball.***: add watermark to baseball notebook
- **notebooks/a_simple_example_8_schools.***: add watermark to notebook
- **notebooks/Gelman_arm_sim_statsmodels.***: add sim statsmodels playground
- **notebooks/Chapter_2_MBML_log_probability.***: add notebook log probability for Chapter 2 MBML
- **notebooks/Chapter_2_MBML_SVI.***: add link to the Pyro forum discussion
- **notebooks/Chapter_2_MBML_numpyro_experiments.***: add numpyro experiments on Chapter 2 MBML
- **notebooks/Chapter_2_MBML_SVI.***: add example of Chapter 2 MBML using SVI
- **notebooks/Bayesian_Naive_Bayes.***: add example of Bayesian Naive Bayes
- **notebooks/Section\-3.6\-of\-Chapter\-3\--\-Inferences\-with\-binomials.***: add example from PyMC3 3.6 Joint distributions from Chapter 3 - Inferences with binomials
- **notebooks/discrete_latent.***: add Predictive to discrete_latent example
- **notebooks/discrete_latent.***: add example for discrete_latent
- **notebooks/Config_Enumerate_in_Pyro.***: add port of config enumerate to project
- **makefile**: add sync target using jupytext
- **notebooks/a_simple_example_8_schools.py-notebooks/a_simple_example_8_schools.ipynb**: add 8 schools example
- **notebooks/baseball.py-notebooks/baseball.ipynb**: add baseball example

### Fix

- **notebooks/Chapter_2_MBML_log_probability.***: fixed the reproduction of Figure 2.29
- **notebooks/Chapter_2_MBML_SVI.***: try TraceGraph_ELBO, add experiments
- **notebooks/Chapter_2_MBML_numpyro_experiments.***: model_05 inference was using model_03
- **notebooks/Section\-3.6\-of\-Chapter\-3\--\-Inferences\-with\-binomials.***: fix the hard coded summary of `n` values for Continous `n` case
- **notebooks/pyro_importance_sampling.***: add more chains, and verify with SA kernel

### Refactor

- **notebooks/Section\-3.6\-of\-Chapter\-3\--\-Inferences\-with\-binomials.***: replace the single use of rng_key with vector of keys

## v0.2.0 (2021-07-02)

### Refactor

- **notebooks/pyro_importance_sampling.ipynb**: attempting to git notebook with output
- **.gitignore**: ignoring everything in scripts, tracking everything in notebooks
- **.gitignore**: refactor notebooks and scripts into seperate dir

### Fix

- **numpyro_play/__init__.py**: import script
- **notebooks/pyro_importance_sampling.py**: fixed the inifinte while loop in model
- **.gitignore**: ignore vscode setting
- **.gitignore**: add gitignore

### Feat

- **numpyro_play/port_ex_impt_sampling.py**: add torch version of simulate from importance sampling
- **notebooks/pyro_importance_sampling.ipynb**: adding notebook for importance sampling
- **notebooks/pyro_importance_sampling.py**: add port of pyro_importance_sampling.py
