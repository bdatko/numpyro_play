## Purpose
A repo of me just experimenting with NumPyro examples.

## Environment 🐢🐢🌱🌿🌵🌎🍀💦
**Note** I am installing numpyro from their GitHub to get latest version

To reproduce the environment you will need to have [`poetry2conda`](https://pypi.org/project/poetry2conda/) installed in you base conda so that the makefile can create the conda  `environment.yaml` from the `pyporject.toml`. Edit the makefile with your paths for your conda root folder.

1. `(base) $ pip install poetry2conda`
2. Edit `conda_root` for your path within the `makefile`
3. `make all`

If you don't want to bother with `make` you can:

1. `conda env create -f environment.yaml`
2. `pip install -q numpyro@git+https://github.com/pyro-ppl/numpyro`
3. `pip install graphviz`

## Notebooks 📓
* explore `notebooks/`
* all notebooks are paired with a python script using `jupytext`

## Wiki
I also have some notes on the [Wiki tab](https://github.com/bdatko/numpyro_play/wiki) 👀 ... s*hhh it's a secret* 🤐
