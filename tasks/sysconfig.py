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
"""Which info pages are org mode"""

RST_DOCS_SOURCES = [
    'glossary',
    'api',
]
"""Which info pages are in raw rst"""

LOGO_DIR = "info/logo"

PYPIRC="$HOME/.pypirc"
TESTING_PYPIRC="$HOME/.pypirc"

# this is the name of the pyenv "version" to use for creating and
# activating conda
PYENV_CONDA_NAME = 'miniconda3-latest'
