"""Testing that the plugin for wepy tests actually works."""
import pytest

fixtures = [
    'lj_sanity_test',
]

def test_lj_plugin(
                   lj_sanity_test,
):

        # sanity test to make sure its even on the path
        assert lj_sanity_test == "sanity"


@pytest.mark.usefixtures(*fixtures)
class TestPlugin():

    def test_fixtures(self):
        pass

    def test_lj_plugin(self,
                       lj_sanity_test,
    ):

        # sanity test to make sure its even on the path
        assert lj_sanity_test == "sanity"

