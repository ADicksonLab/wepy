"""Configuration managed by the system. All changes here will be
overwrote upon update.

Typically gives a collection of good defaults. Override in config.py

"""

### Cleaning

CLEAN_EXPRESSIONS = [
    "\"*~\"",
]


### Envs

# which virtual environment tool to use: venv or conda
ENV_METHOD = 'venv'

# which env spec to use by default
DEFAULT_ENV = 'dev'

# directory where env specs are read from
ENVS_DIR = 'envs'

# Python version source, this is how we get the different python
# versions. This is a keyword not a path
PYTHON_VERSION_SOURCE = "pyenv"

# which versions will be requested to be installed, in the order of
# precendence for interactive work
PYTHON_VERSIONS = (
    '3.8.1',
    '3.7.6',
    '3.6.10',
    'pypy3.6-7.3.0',
)


### Git

INITIAL_VERSION = '0.0.0a0.dev0'
GIT_LFS_TARGETS = []
VERSION = '0.0.0a0.dev0'


### Python Code base

REPORTS_DIR = "reports"

## docs

LOGO_DIR = "info/logo"

ORG_DOCS_SOURCES = [
    'changelog',
    'dev_guide',
    'general_info',
    'installation',
    'introduction',
    'news',
    'quick_start',
    'troubleshooting',
    'users_guide',
    'reference',
]

RST_DOCS_SOURCES = [
    'glossary',
    'api',
]

BIB_DOCS_SOURCES = [
    'docs',
]

PYPIRC="$HOME/.pypirc"
TESTING_PYPIRC="$HOME/.pypirc"

# this is the name of the pyenv "version" to use for creating and
# activating conda
PYENV_CONDA_NAME = 'miniconda3-latest'

## tests
TESTS_DIR = "tests"

## benchmarks
BENCHMARKS_DIR = "benchmarks"

# the range of commits to use for running all of the asv regression
# benchmarks. See documentation in `asv run --help` for details.
# Defaults to using the HASHFILE.
ASV_RANGE = "HASHFILE:benchmark_selection.list"


### Containers

# choose the container tool, options really are just: docker or podman
# since these two are compatible
CONTAINER_TOOL = "podman"
