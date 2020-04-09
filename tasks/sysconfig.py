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

### Git

INITIAL_VERSION = '0.0.0a0.dev0'
GIT_LFS_TARGETS = []
VERSION = '0.0.0a0.dev0'

### Python Code base

BENCHMARK_STORAGE_URL="./metrics/benchmarks"
"""Directory to store benchmarks of code"""

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
    'tutorials/index',
]
"""Which info pages are in raw rst"""

PYPIRC="$HOME/.pypirc"
TESTING_PYPIRC="$HOME/.pypirc"
