# Makefile
SHELL = /bin/bash
VENV = .venv
PYTHON_VERSION = 3.10.9

# Colors for echos
ccend=$(shell tput sgr0)
ccso=$(shell tput smso)

python:
	pyenv install ${PYTHON_VERSION}

.ONESHELL:
env:
	@echo ""
	@echo "$(ccso)--> Creating virtual environment $(ccend)"
	pyenv local ${PYTHON_VERSION}
	poetry install
	poetry shell
	pre-commit install && \
	pre-commit autoupdate

.ONESHELL:
delete-env:
	@echo ""
	@echo "$(ccso)--> Removing virtual environment $(ccend)"
	rm -rf ${VENV}
	@if [ -f .python-version ]; then\
		rm .python-version;\
	fi

.ONESHELL:
style:
	black .
	flake8
	isort .

.ONESHELL:
clean:
	find . -type f -name "*.DS_Store" -ls -delete
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E ".pytest_cache" | xargs rm -rf
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf
	find . | grep -E ".trash" | xargs rm -rf
	rm -f .coverage

.ONESHELL:
dvc:
	poetry shell
	dvc add data/raw/ml-engineer-challenge-redacted-data.csv
	dvc add data/data_validation/.
	dvc add data/preprocessed_labels/.
	dvc push
