from invoke import task

from ..config import (
    VERSION,
    REPORTS_DIR,
    ORG_DOCS_SOURCES,
    RST_DOCS_SOURCES,
    BIB_DOCS_SOURCES,
    LOGO_DIR,
    TESTING_PYPIRC,
    PYPIRC,
    PYENV_CONDA_NAME,
    ENV_METHOD,
    TESTS_DIR,
    BENCHMARKS_DIR,
)

import sys
import os
import os.path as osp
from pathlib import Path
import shutil as sh

## User config examples

# SNIPPET:
# REPORTS_DIR = "reports"
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

# PYENV_CONDA_NAME = 'miniconda3-latest'

## CONSTANTS


BENCHMARK_STORAGE_URI = f"\"file://{REPORTS_DIR}/benchmarks\""


def project_slug():

    try:
        from ..config import PROJECT_SLUG
    except ImportError:
        print("You must set the 'PROJECT_SLUG' in conifig.py to use this")
    else:
        return PROJECT_SLUG

@task
def init(cx):

    # install the versioneer files

    # this always exits in an annoying way so we just warn here.
    cx.run("versioneer install",
           warn=True)

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
    cx.run("rm -rf sphinx/_build")
    cx.run("rm -rf sphinx/_source")
    cx.run("rm -rf sphinx/_api")
    cx.run("rm -rf sphinx/_static")

    # reports
    cx.run("rm -rf reports/benchmarks/asv/_html")

@task
def docs_regressions(cx):

    with cx.cd("benchmarks"):
        cx.run("asv publish", warn=True)

@task
def docs_coverage(cx):
    cx.run("coverage html -d reports/coverage/_html/index.html",
           warn=True)

@task
def docs_complexity(cx):

    os.makedirs(f"{REPORTS_DIR}/code_quality/_html",
                exist_ok=True)
    cx.run(f"lizard -o {REPORTS_DIR}/code_quality/_html/index.html src/{project_slug()}",
           warn=True)

@task(pre=[
    docs_regressions,
    docs_coverage,
    docs_complexity,
])
def docs_reports(cx):
    """Build all of the reports from source."""
    pass

@task(pre=[docs_clean, docs_reports])
def docs_build(cx):
    """Buld the documentation"""

    # make sure the 'source' folder exists
    cx.run("mkdir -p sphinx/_source")
    cx.run("mkdir -p sphinx/_source/tutorials")
    cx.run("mkdir -p sphinx/_source/examples")
    cx.run("mkdir -p sphinx/_static")

    # copy the logo over
    cx.run(f"cp {LOGO_DIR}/* sphinx/_static/")

    # and the other theming things
    cx.run(f"cp sphinx/static/* sphinx/_static/")

    # copy the plain RST files over to the sources
    for source in RST_DOCS_SOURCES:
        cx.run(f"cp info/{source}.rst sphinx/_source/{source}.rst")

    # copy the Bibtex files over
    for source in BIB_DOCS_SOURCES:
        cx.run(f"cp info/{source}.bib sphinx/_source/{source}.bib")

    # convert the org mode to rst in the source folder
    for source in ORG_DOCS_SOURCES:

        # name it the same
        target = source

        cx.run("pandoc "
               "-f org "
               "-t rst "
               f"-o sphinx/_source/{target}.rst "
               f"info/{source}.org")

    ## Examples

    # examples don't get put into the documentation and rendered like
    # the tutorials do, but we do copy the README as the index

    # copy the tutorials_index.rst file to the tutorials in _source
    # TODO: remove, don't think I will use this
    # sh.copyfile(
    #     "sphinx/examples_index.rst",
    #     "sphinx/_source/examples/index.rst",
    # )


    ## Tutorials

    # convert the README
    sh.copyfile(
        "sphinx/tutorials_index.rst",
        "sphinx/_source/tutorials/index.rst",
    )

    # convert any of the tutorials that exist with an org mode extension as well
    for item in os.listdir('info/tutorials'):
        item = Path('info/tutorials') / item

        # tutorials are in their own dirs
        if item.is_dir():

            docs = list(item.glob("README.org")) + \
                list(item.glob("README.ipynb")) + \
                list(item.glob("README.rst"))

            if len(docs) > 1:
                raise ValueError(f"Multiple tutorial files for {item}")
            else:
                readme_path = docs[0]

            tutorial = item.stem

            os.makedirs(f"sphinx/_source/tutorials/{tutorial}",
                        exist_ok=True)

            # we must convert org mode files to rst
            if readme_path.suffix == '.org':

                cx.run("pandoc "
                       "-f org "
                       "-t rst "
                       f"-o sphinx/_source/tutorials/{tutorial}/README.rst "
                       f"info/tutorials/{tutorial}/README.org")

            # just copy notebooks since teh sphinx extension handles
            # them
            elif readme_path.suffix in ('.ipynb', '.rst',):

                sh.copyfile(
                    readme_path,
                    f"sphinx/_source/tutorials/{tutorial}/{readme_path.stem}{readme_path.suffix}",
                )

            # otherwise just move it
            else:
                raise ValueError(f"Unkown tutorial type for file: {readme_path.stem}{readme_path.suffix}")


    # run the build steps for sphinx
    with cx.cd('sphinx'):

        # build the API Documentation
        cx.run(f"sphinx-apidoc -f --separate --private --ext-autodoc --module-first --maxdepth 1 -o _api ../src/{project_slug()}")

        # then do the sphinx build process
        cx.run("sphinx-build -b html -E -a -j 6 -c . . ./_build/html/")



    ## Post Sphinx

    # add things like adding in metrics etc. here
    # copy the benchmark regressions over if available

    # asv regressions
    regression_pages = Path("reports/benchmarks/asv/_html")

    if regression_pages.exists() and regression_pages.is_dir():
        sh.copytree(
            regression_pages,
            "sphinx/_build/html/regressions"
        )

    quality_pages = Path("reports/code_quality/_html")

    if quality_pages.exists() and quality_pages.is_dir():
        sh.copytree(
            quality_pages,
            "sphinx/_build/html/quality"
        )

    # coverage
    coverage_pages = Path("reports/coverage/_html")

    if coverage_pages.exists() and coverage_pages.is_dir():
        sh.copytree(
            coverage_pages,
            "sphinx/_build/html/coverage"
        )





@task(pre=[docs_build])
def docs_serve(cx):
    """Local server for documenation"""
    cx.run("python -m http.server -d sphinx/_build/html 8022")

### Website

@task(pre=[clean_docs, clean_website, docs_build])
def website_serve(cx):
    """Serve the main web page locally for development."""

    # TODO: implement using Nikola

    # STUB: just use the docs for this right now
    docs_serve(cx)


# STUB: @task(pre=[clean_docs, docs_build])
@task
def website_deploy(cx):
    """Deploy the documentation onto the internet."""

    # use the ghp-import tool which handles the branch switching to
    # `gh-pages` for you
    cx.run("ghp-import --no-jekyll --push --force sphinx/_build/html")


### Jigs

# jigs are for that kind of in between work of not in module, not
# documentation etc. Could be prototypes, troubleshooting, or anything
# that needs non-trivial setup but isn't part of a "framework". Uses
# the same schema as examples to give some order to it.

@task
def new_jig(cx, name=None, template="org", env='venv_blank'):
    """Create a new jig.

    Can choose between the following templates:

    - 'org' :: org mode notebook

    Choose from the following env templates:

    - None
    - venv_blank
    - venv_dev
    - conda_blank
    - conda_dev

    """

    assert name is not None, "Must provide a name"

    template_path = Path(f"templates/jigs/{template}")

    # check if the template exists
    if not template_path.is_dir():

        raise ValueError(
            f"Unkown template {template}. Check the 'templates/jigs' folder")

    # check if the env exists
    if env is not None:
        env_tmpl_path = Path(f"templates/envs/{env}")

        if not env_tmpl_path.is_dir():

            raise ValueError(
                f"Unkown env template {env}. Check the 'templates/envs' folder")


    target_path = Path(f"jigs/{name}")

    if target_path.exists():
        raise FileExistsError(f"Jig with name {name} already exists. Not overwriting.")

    # copy the template
    cx.run(f"cp -r {template_path} {target_path}")

    # copy the env
    cx.run(f"cp -r {env_tmpl_path} {target_path / 'env'}")

    print(f"New example created at: {target_path}")

@task
def pin_jig(cx, name=None):
    """Pin the deps for an example or all of them if 'name' is None."""

    path = Path('jigs') / name / 'env'

    assert path.exists() and path.is_dir(), \
        f"Env for Jig {name} doesn't exist"

    cx.run(f"inv env.deps-pin-path -p {path}")

@task
def env_jig(cx, name=None):
    """Make a the example env in its local dir."""

    if name is None:
        raise ValueError("Must specify which jig to use")

    spec_path = Path('jigs') / name / 'env'
    env_path = Path('jigs') / name / '_env'

    assert spec_path.exists() and spec_path.is_dir(), \
        f"Jig {name} doesn't exist"

    cx.run(f"inv env.make-env -s {spec_path} -p {env_path}")


### Tests


@task
def tests_benchmarks(cx):
    cx.run("pytest -m 'not interactive' tests/test_benchmarks",
           warn=True)

@task
def tests_integration(cx, tag=None):

    if tag is None:
        cx.run(f"coverage run -m pytest -m 'not interactive' tests/test_integration",
               warn=True)
    else:
        cx.run(f"coverage run -m pytest --html=reports/pytest/{tag}/integration/report.html -m 'not interactive' tests/test_integration",
               warn=True)


@task
def tests_unit(cx, tag=None):

    if tag is None:
        cx.run(f"coverage run -m pytest -m 'not interactive' tests/test_unit",
               warn=True)
    else:
        cx.run(f"coverage run -m pytest --html=reports/pytest/{tag}/unit/report.html -m 'not interactive' tests/test_unit",
               warn=True)


@task
def tests_interactive(cx):
    """Run the interactive tests so we can play with things."""
    cx.run("pytest -m 'interactive'",
           warn=True)

@task()
def tests_all(cx, tag=None):
    """Run all the automated tests. No benchmarks.

    There are different kinds of nodes that we can run on that
    different kinds of tests are available for.

    - minor : does not have a GPU, can still test most other code paths

    - dev : has at least 1 GPU, enough for small tests of all code paths

    - production : has multiple GPUs, good for running benchmarks
                   and full stress tests

    """

    tests_unit(cx, tag=tag)
    tests_integration(cx, tag=tag)

@task
def tests_nox(cx):

    if ENV_METHOD == 'pyenv':

        # run with base venv maker
        with cx.prefix("unset PYENV_VERSION"):
            cx.run("nox -s test")

    elif ENV_METHOD == 'conda':

        # test running with conda
        with cx.prefix(f"pyenv shell {PYENV_CONDA_NAME}"):
            cx.run("nox -s test")

    else:

        raise ValueError(f"Unsupported ENV_METHOD: {ENV_METHOD}")


### Code & Test Quality

@task
def docstrings_report(cx):

    cx.run("mkdir -p reports/docstring_coverage")
    cx.run("interrogate -o reports/docstring_coverage/src.interrogate.txt -vv src")

    # TODO add it for tests and docs etc.

@task
def coverage_report(cx):
    # cx.run("coverage report")
    cx.run("coverage xml -o reports/coverage/coverage.xml",
           warn=True)
    cx.run("coverage json -o reports/coverage/coverage.json",
           warn=True
    )

@task
def coverage_serve(cx):
    cx.run("python -m http.server -d reports/coverage/html 8020",
           asynchronous=True)


@task
def lint(cx):

    cx.run(f"mkdir -p {REPORTS_DIR}/lint")

    cx.run(f"rm -f {REPORTS_DIR}/lint/flake8.txt")
    cx.run(f"flake8 --output-file={REPORTS_DIR}/lint/flake8.txt src/{project_slug()}",
           warn=True)

@task
def complexity(cx):
    """Analyze the complexity of the project."""

    cx.run(f"mkdir -p {REPORTS_DIR}/code_quality")

    cx.run(f"lizard -o {REPORTS_DIR}/code_quality/lizard.csv src/{project_slug()}",
           warn=True)

    # SNIPPET: annoyingly opens the browser

    # make a cute word cloud of the things used
    # cx.run(f"(cd {REPORTS_DIR}/code_quality; lizard -EWordCount src/project_slug() > /dev/null)")

@task
def complexity_serve(cx):
    cx.run("python -m http.server -d reports/conde_quality/lizard.html 8021",
           asynchronous=True)

@task(pre=[coverage_report, complexity, lint])
def quality(cx):
    pass

@task(pre=[coverage_serve, complexity_serve])
def quality_serve(cx):
    pass


### Profiling

@task
def profile(cx):
    NotImplemented

### Performance Benchmarks

## regressions

@task
def regressions_all(cx):
    """Run regression benchmarks for all of the hashes/tags in the
    benchmarks/benchmark_selection.list file"""

    with cx.cd("benchmarks"):
        cx.run("asv run HASHFILE:benchmark_selection.list")

@task
def regression_current(cx):

    with cx.cd("benchmarks"):
        cx.run("asv run")

# @task
# def asv_update

@task
def benchmark_adhoc(cx):
    """An ad hoc benchmark that will not be saved."""

    cx.run("pytest benchmarks/pytest_benchmark/test_benchmarks")

@task
def benchmark_save(cx):
    """Run a proper benchmark that will be saved into the metrics for regression testing etc."""

    run_command = \
f"""pytest --benchmark-autosave --benchmark-save-data \
          --benchmark-storage={BENCHMARK_STORAGE_URI} \
          tests/test_benchmarks
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

    # WKRD, FIXME: because new versions of pip are incompatible with
    # pip-tools we can't blindly update in envs. Want to make a
    # mechanism for installing the tools in a separate
    # requirements.txt file since we can't put these in the pip tools
    # input files. In short should be taken care of in the 'env' module
    print("Disable 'update_tools' use the envs 'tools.requirements.txt' instead")
    pass
    # cx.run("pip install --upgrade pip setuptools wheel twine")

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
