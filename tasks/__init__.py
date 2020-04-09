from invoke import Collection, Task, task

import inspect


## Utilities

# these helper functions are for automatically listing all of the
# functions defined in the tasks module

def _is_mod_task(mod, func):
    return issubclass(type(func), Task) and inspect.getmodule(func) == mod

def _get_functions(mod):
    """get only the functions that aren't module functions and that
    aren't private (i.e. start with a '_')"""

    return {func.__name__ : func for func in mod.__dict__.values()
            if _is_mod_task(mod, func) }


## Namespace

# add all of the modules to the CLI
ns = Collection()

## Top-level

from . import toplevel
for func in _get_functions(toplevel).values():
    ns.add_task(func)

## Upstream

from .modules import MODULES as modules

for module in modules:
    ns.add_collection(module)

## Plugins

try:
    # import all the user defined stuff and override
    from .plugins import PLUGIN_MODULES as plugins

    for module in plugins:
        ns.add_collection(module)

except Exception as e:
    print("Loading plugins failed with error ignoring:")
    print(e)
