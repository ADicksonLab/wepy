"""Specify which plugins to load"""

# import plugins:

from . import custom

# specify which plugins to install, the custom one is included by
# default to get users going
PLUGIN_MODULES = [
    custom,
]
