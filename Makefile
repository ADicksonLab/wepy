## Notes on this file. The `help` target parses the file and can generate
## headings and docstrings for targets, in the order they are in the file. To
## create a heading make a comment like `##@ Heading Content`. To document
## targets make a comment on the same line as the target name with two `##
## docstring explanation...`. To leave a target undocumented simply provide no
## docstring.

PYTHON=python3.11
TMPDIR=.tmp

##@ Getting Started

env: ## Rebuild the main development environment
	nox -s dev_external
	echo "Run this to activate dev env: . env.sh"
.PHONY: env

##@ Housekeeping

clean-all: clean clean-docs clean-env clean-compose clean-hooks ## run all clean commands
.PHONY: clean-all

clean: ## Clean temporary files, directories etc.
	rm -rf $(TMPDIR)
	rm -rf dist .pytest_cache .mypy_cache .coverage .venv .nox .hatch htmlcov
	find . -type f -name "*.pyc" -print -delete
	hatch clean
.PHONY: clean

clean-env: ## Clean the dev environments
	rm -rf .venv
.PHONY: clean-env

##@ QA (TODO)

# format: ## Run source code formatters manually.
# 	nox -s format
# .PHONY: docstrings

# validate:  ## Run all linters, type checks, static analysis, etc.
# 	nox -s validate
# .PHONY: validate

# format-check: ## Run code formatting checks
# 	nox -s format_check
# .PHONY: format-check

# # check: ## Run only miscellaneous maintenance checks
# # .PHONY: check

# lint: ## Run only the linters (non-autoformatters).
# 	nox -s lint
# .PHONY: lint

# docstring-check: ## Run docstring coverage only.
# 	nox -s docstring_lint
# .PHONY: docstring

# typecheck: ## Run only the type checker (requires mypy)
# 	nox -s typecheck
# .PHONY: typecheck

##@ Dev

test-unit: ## Run unit tests with coverage report
	nox -s tests_unit
.PHONY: test

coverage: ## Report on missing coverage. (Run 'test-unit' to generate new stats)
	nox -s coverage
.PHONY: coverage

serve-coverage: ## Run a temporary web server to display detailed coverage report
	python3 -m http.server --directory htmlcov 4322
.PHONY: serve-coverage


##@ Documentation (TODO)

# docs: ## TODO: Build the documentation
# 	nox -s docs
# .PHONY: docs

# clean-docs: ## TODO: Clean temporary files for documentation
# 	rm -rf docs/_api docs/_build
# .PHONY: clean-docs

# serve-docs: ## TODO: Run a temporary web server to display current documentation build
# 	python3 -m http.server --directory docs/_build 4323
# .PHONY: serve-docs


##@ Release Management

pin: pyproject.toml ## Pin the project dependency versions
	nox -s pin
.PHONY: pin

bumpversion: ## Bump the minor version for the project
	nox -s bumpversion
.PHONY: bumpversion

build: ## Run the python build/packaging, generate sdist & wheel
	nox -s build
.PHONY: build

publish: ## Publish the package to indices
	nox -s publish
.PHONY: publish

##@ Help

# An automatic help command: https://www.padok.fr/en/blog/beautiful-makefile-awk
.DEFAULT_GOAL := help

help: ## (DEFAULT) This command, show the help message
	@echo "See CONTRIBUTING.md for dependencies, then run this:"
	@echo ""
	@echo "If you want a shell in a virtual environment with everything:"
	@echo "  > make env"
	@echo "  > . ./env.sh"
	@echo ""
	@echo "Do testing:"
	@echo "  > make test-unit"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)
.PHONY: help
