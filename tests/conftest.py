"""Root level conftest.

This provides fixtures and adds the utility importable modules
available to all tests.

If there are domain specific modules put them in the correct folders.

"""
import sys
import os
import os.path as osp
from pathlib import Path

tests_dir = Path(osp.dirname(__file__))
utils_dir = tests_dir / 'utils'

sys.path.append(str(utils_dir))
