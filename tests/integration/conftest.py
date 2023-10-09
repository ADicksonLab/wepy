# Third Party Library
import pytest


# using this to get rid of the warning without having to put it in my
# config file
def pytest_configure(config):
    config.addinivalue_line("markers", "interactive: tests which give you the debugger")
