
# SNIPPET: add this to import modules

# should be copied in by the installation process
from . import core
from . import clean
from . import env
from . import git
from . import py
from . import docs
from . import lxd
from . import containers

MODULES = [
    core,
    clean,
    env,
    git,
    py,
    docs,
    lxd,
    containers,
]
