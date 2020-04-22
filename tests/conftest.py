"""Root level conftest.

This provides fixtures and adds the utility importable modules
available to all tests.

If there are domain specific modules put them in the correct folders.

"""
import sys
import os
import os.path as osp
from pathlib import Path

import pytest

pytest_plugins = [
    'pytest_wepy.lennard_jones_pair',
]


# NOTE: not used much now, but this is the way to do this properly
tests_dir = Path(osp.dirname(__file__))
utils_dir = tests_dir / 'utils'

sys.path.append(str(utils_dir))
