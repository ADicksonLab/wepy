"""Generic modules that help with running tests more smoothly."""

# Standard Library
import os
import os.path as osp
from contextlib import contextmanager


@contextmanager
def cd(newdir):
    """Change directories use as a context manager."""
    prevdir = os.getcwd()
    os.chdir(osp.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)
