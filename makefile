SHELL=/usr/bin/bash
.DEFAULT_GOAL := help

env_name:=numpyro_play
conda:=conda
fixtures:=tests/fixtures/
pytestflags:=

conda_root := /home/benda/anaconda3/bin
conda_activate := $(conda_root)/activate
activate_env := source $(conda_activate) $(env_name)

.PHONY: all
## Make the project
all: .cenv

## Create the environment.yaml file from poetry2conda
environment.yaml: pyproject.toml
	poetry2conda --dev pyproject.toml environment.yaml

# https://stackoverflow.com/a/61410041/3587374

## Create the conda environment from the `environment.yaml` you have to activate afterwards
.cenv: environment.yaml
	$(conda) env create --quiet --force --file environment.yaml; \
	touch .cenv
	@echo to activate: conda activate $(env_name)

.PHONY: test
## Test for source and activate
test:
	@echo $(activate_env) && python --version 


.PHONY: realclean
## Remove unused conda packages then remove the environment
realclean:
	$(conda) clean -a
	$(conda) env remove --name $(env_name)
	rm -f .cenv

# Adopted from https://gist.github.com/klmr/575726c7e05d8780505a
# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: show-help
show-help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) == Darwin && echo '--no-init --raw-control-chars')
