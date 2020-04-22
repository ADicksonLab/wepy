
import pytest

@pytest.fixture(scope='class')
def test_wepy_fixture():

    return "Hello"
