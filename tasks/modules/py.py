from invoke import task

from ..config import (
    VERSION,
    BENCHMARK_STORAGE_URL,
    ORG_DOCS_SOURCES,
    RST_DOCS_SOURCES,
    TESTING_PYPIRC,
    PYPIRC,
)

import sys
import os
import os.path as osp
from pathlib import Path

## User config examples

# SNIPPET:
# BENCHMARK_STORAGE_URL="./metrics/benchmarks"
# ORG_DOCS_SOURCES = [
#     'changelog',
#     'dev_guide',
#     'general_info',
#     'installation',
#     'introduction',
#     'news',
#     'quick_start',
#     'troubleshooting',
#     'users_guide',
#     'reference',
# ]
# RST_DOCS_SOURCES = [
#     'glossary',
#     'tutorials/index',
# ]
# PYPIRC = "$HOME/.pypirc"
# TESTING_PYPIRC = "$HOME/.test-pypirc"

## CONSTANTS


BENCHMARK_STORAGE_URI="\"file://{}\"".format(BENCHMARK_STORAGE_URL)


def project_slug():

    try:
        from ..config import PROJECT_SLUG
    except ImportError:
        print("You must set the 'PROJECT_SLUG' in conifig.py to use this")
    else:
        return PROJECT_SLUG

@task
def clean_dist(cx):
    """Remove all build products."""

    cx.run("python setup.py clean")
    cx.run("rm -rf dist build */*.egg-info *.egg-info")

@task
def clean_cache(cx):
    """Remove all of the __pycache__ files in the packages."""
    cx.run('find . -name "__pycache__" -exec rm -r {} +')

@task
def clean_docs(cx):
    """Remove all documentation build products"""

    docs_clean(cx)

@task
def clean_website(cx):
    """Remove all local website build products"""
    cx.run("rm -rf docs/*")

    # if the website accidentally got onto the main branch we remove
    # that crap too
    for thing in [
            '_images',
            '_modules',
            '_sources',
            '_static',
            'api',
            'genindex.html',
            'index.html',
            'invoke.html',
            'objects.inv',
            'py-modindex.html',
            'search.html',
            'searchindex.js',
            'source',
            'tutorials',
    ]:

        cx.run(f"rm -rf {thing}")

@task(pre=[clean_cache, clean_dist, clean_docs, clean_website])
def clean(cx):
    pass


### Docs

@task
def docs_clean(cx):

    cx.run("cd sphinx && make clean")
    cx.run("rm -rf sphinx/_build/*")
    cx.run("rm -rf sphinx/source/*")
    cx.run("rm -rf sphinx/api/*")

@task(pre=[docs_clean,])
def docs_build(cx):
    """Buld the documenation"""

    # make sure the 'source' folder exists
    cx.run("mkdir -p sphinx/source")
    cx.run("mkdir -p sphinx/source/tutorials")

    # copy the plain RST files over to the sources
    for source in RST_DOCS_SOURCES:
        cx.run(f"cp info/{source}.rst sphinx/source/{source}.rst")

    # convert the org mode to rst in the source folder
    for source in ORG_DOCS_SOURCES:

        # name it the same
        target = source

        cx.run("pandoc "
               "-f org "
               "-t rst "
               f"-o sphinx/source/{target}.rst "
               f"info/{source}.org")

    # convert any of the tutorials that exist with an org mode extension as well
    for source in os.listdir('info/tutorials'):
        source = Path(source)

        # if it is org mode convert it and put it into the sources
        if source.suffix == '.org':

            source = source.stem
            cx.run("pandoc "
                   "-f org "
                   "-t rst "
                   f"-o sphinx/source/tutorials/{source}.rst "
                   f"info/tutorials/{source}.org")

        # otherwise just move it
        else:

            # quick check for other kinds of supported files
            assert source.suffix in ['.ipynb', '.rst',]

            cx.run(f"cp info/tutorials/{source} sphinx/source/tutorials/{source}")

    # run the build steps for sphinx
    with cx.cd('sphinx'):

        # build the API Documentation
        cx.run(f"sphinx-apidoc -f --separate --private --ext-autodoc --module-first --maxdepth 1 -o api ../src/{project_slug()}")

        # then do the sphinx build process
        cx.run("sphinx-build -b html -E -a -j 6 . ./_build/html/")


@task(pre=[docs_build])
def docs_serve(cx):
    """Local server for documenation"""
    cx.run("python -m http.server -d sphinx/_build/html")

### TODO: WIP Website

@task(pre=[clean_docs, clean_website, docs_build])
def website_deploy_local(cx):
    """Deploy the docs locally for development. Must have bundler and jekyll installed"""


    cx.cd("jekyll")

    # update dependencies
    cx.run("bundle install")
    cx.run("bundle update")

    # run the server
    cx.run("bundle exec jekyll serve")

# STUB: @task(pre=[clean_docs, docs_build])
@task
def website_deploy(cx):
    """Deploy the documentation onto the internet."""

    cx.run("(cd sphinx; ./deploy.sh)")



### Tests


@task
def tests_benchmarks(cx):
    cx.run("(cd tests/tests/test_benchmarks && pytest -m 'not interactive')")

@task
def tests_integration(cx):
    cx.run(f"(cd tests/tests/test_integration && pytest -m 'not interactive')")

@task
def tests_unit(cx):
    cx.run(f"(cd tests/tests/test_unit && pytest -m 'not interactive')")

@task
def tests_interactive(cx):
    """Run the interactive tests so we can play with things."""

    cx.run("pytest -m 'interactive'")

@task()
def tests_all(cx):
    """Run all the automated tests. No benchmarks.

    There are different kinds of nodes that we can run on that
    different kinds of tests are available for.

    - minor : does not have a GPU, can still test most other code paths

    - dev : has at least 1 GPU, enough for small tests of all code paths

    - production : has multiple GPUs, good for running benchmarks
                   and full stress tests

    """


    tests_unit(cx)
    tests_integration(cx)

@task
def tests_tox(cx):

    NotImplemented

    TOX_PYTHON_DIR=None

    cx.run("env PATH=\"{}/bin:$PATH\" tox".format(
        TOX_PYTHON_DIR))

### Code Quality

@task
def lint(cx):

    cx.run("mkdir -p metrics/lint")

    cx.run("rm -f metrics/lint/flake8.txt")
    cx.run(f"flake8 --output-file=metrics/lint/flake8.txt src/{project_slug()}")

@task
def complexity(cx):
    """Analyze the complexity of the project."""

    cx.run("mkdir -p metrics/code_quality")

    cx.run(f"lizard -o metrics/code_quality/lizard.csv src/{project_slug()}")
    cx.run(f"lizard -o metrics/code_quality/lizard.html src/{project_slug()}")

    # SNIPPET: annoyingly opens the browser

    # make a cute word cloud of the things used
    # cx.run("(cd metrics/code_quality; lizard -EWordCount src/project_slug() > /dev/null)")

@task(pre=[complexity, lint])
def quality(cx):
    pass


### Profiling and Performance

@task
def profile(cx):
    NotImplemented

@task
def benchmark_adhoc(cx):
    """An ad hoc benchmark that will not be saved."""

    cx.run("pytest tests/tests/test_benchmarks")

@task
def benchmark_save(cx):
    """Run a proper benchmark that will be saved into the metrics for regression testing etc."""

    run_command = \
f"""pytest --benchmark-autosave --benchmark-save-data \
          --benchmark-storage={BENCHMARK_STORAGE_URI} \
          tests/tests/test_benchmarks
"""

    cx.run(run_command)

@task
def benchmark_compare(cx):

    # TODO logic for comparing across the last two

    run_command = \
"""pytest-benchmark \
                    --storage {storage} \
                    compare 'Linux-CPython-3.6-64bit/*' \
                    --csv=\"{csv}\" \
                    > {output}
""".format(storage=BENCHMARK_STORAGE_URI,
           csv="{}/Linux-CPython-3.6-64bit/comparison.csv".format(BENCHMARK_STORAGE_URL),
           output="{}/Linux-CPython-3.6-64bit/report.pytest.txt".format(BENCHMARK_STORAGE_URL),
)

    cx.run(run_command)

@task
def version_which(cx):
    """Tell me what version the project is at."""

    # get the current version
    cx.run(f"python -m {project_slug()}._print_version")

### Packaging

## Building

# IDEA here are some ideas I want to do

# Source Distribution

# Wheel: Binary Distribution

# Beeware cross-patform

# Debian Package (with `dh_virtualenv`)

@task
def update_tools(cx):
    cx.run("pip install --upgrade pip setuptools wheel twine")

@task(pre=[update_tools])
def build_sdist(cx):
    """Make a source distribution"""
    cx.run("python setup.py sdist")

@task(pre=[update_tools])
def build_bdist(cx):
    """Make a binary wheel distribution."""

    cx.run("python setup.py bdist_wheel")

# STUB
@task
def conda_build(cx):

    cx.run("conda-build conda-recipe")

@task(pre=[build_sdist, build_bdist,])
def build(cx):
    """Build all the python distributions supported."""
    pass


# IDEA: add a 'test_builds' target, that opens a clean environment and
# installs each build


## Publishing

# testing publishing



@task(pre=[clean_dist, build_sdist])
def publish_test_pypi(cx,
                      version=None,
):

    assert version is not None

    cx.run("twine upload "
           "--non-interactive "
           f"--repository pypi_test "
           f"--config-file {TESTING_PYPIRC} "
           "dist/*")

@task(pre=[clean_dist, update_tools, build_sdist])
def publish_test(cx):

    publish_test_pypi(cx,
                      version=VERSION)

# PyPI


@task(pre=[clean_dist, build])
def publish_pypi(cx, version=None):

    assert version is not None

    cx.run(f"twine upload "
           "--non-interactive "
           "--repository pypi "
           f"--config-file {PYPIRC} "
           f"dist/*")


# TODO, SNIPPET, STUB: this is a desired target for uploading dists to github
# @task
# def publish_github_dists(cx, release=None):
#     assert release is not None, "Release tag string must be given"
#     pass

@task(pre=[clean_dist, update_tools, build])
def publish(cx):

    publish_pypi(cx, version=VERSION)
