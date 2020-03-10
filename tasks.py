from invoke import task

import sys
import os
import os.path as osp
from pathlib import Path

BENCHMARK_STORAGE_URL="./metrics/benchmarks"
BENCHMARK_STORAGE_URI="\"file://{}\"".format(BENCHMARK_STORAGE_URL)

ENV_METHOD = 'venv'

# directories the actual environments are stored
VENV_DIR = "_venv"
CONDA_ENVS_DIR = "_conda_envs"

SELF_REQUIREMENTS = 'self.requirements.txt'
PYTHON_VERSION_FILE = 'py_version.txt'

VCS_RELEASE_TAG_TEMPLATE = "v{}"

DEFAULT_ENV = 'dev'

### Version Control

@task
def vcs_init(cx):

    initial_version = "2020-03-09a0.dev0"
    tag_string = VCS_RELEASE_TAG_TEMPLATE.format(initial_version)

    cx.run("git init && "
           "git add -A && "
           "git commit -m 'initial commit' && "
           f"git tag -a {tag_string} -m 'initialization release'")

### Environments

def conda_env(cx, name=DEFAULT_ENV):

    # locally scoped since the environment is global to the
    # anaconda installation
    env_name = name

    # where the specs of the environment are
    env_spec_path = Path("envs") / name

    # using the local envs dir
    env_dir = Path(CONDA_ENVS_DIR) / name

    # figure out which python version to use, if the 'py_version.txt'
    # file exists read it
    py_version_path = env_spec_path / PYTHON_VERSION_FILE
    if osp.exists(py_version_path):
        with open(py_version_path, 'r') as rf:
            py_version = rf.read().strip()

        # TODO: validate the string for python version

    # otherwise use the one you are currently using
    else:
        py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    # create the environment
    cx.run(f"conda create -y "
           f"--prefix {env_dir} "
           f"python={py_version}",
        pty=True)

    with cx.prefix(f'eval "$(conda shell.bash hook)" && conda activate {env_dir}'):

        # install the conda dependencies. choose a specification file
        # based on these priorities of most pinned to least frozen.
        if osp.exists(env_spec_path / "env.pinned.yaml"):

            cx.run(f"conda env update "
                   f"--prefix {env_dir} "
                   f"--file {env_spec_path}/env.pinned.yaml")


        elif osp.exists(env_spec_path / "env.yaml"):

            cx.run(f"conda env update "
                   f"--prefix {env_dir} "
                   f"--file {env_spec_path}/env.yaml")

        else:
            # don't do a conda env pin
            pass


        # install the extra pip dependencies
        if osp.exists(f"{env_spec_path}/requirements.txt"):
            cx.run(f"{env_dir}/bin/pip install "
                   f"-r {env_spec_path}/requirements.txt")

        # install the package itself
        if osp.exists(f"{env_spec_path}/self.requirements.txt"):
            cx.run(f"{env_dir}/bin/pip install -r {env_spec_path}/self.requirements.txt")

    print("--------------------------------------------------------------------------------")
    print(f"run: conda activate {env_dir}")


def venv_env(cx, name=DEFAULT_ENV):

    venv_dir_path = Path(VENV_DIR)
    venv_path = venv_dir_path / name

    env_spec_path = Path("envs") / name

    # ensure the directory
    cx.run(f"mkdir -p {venv_dir_path}")

    # create the env requested
    cx.run(f"python -m venv {venv_path}")

    # then install the things we need
    with cx.prefix(f"source {venv_path}/bin/activate"):

        if osp.exists(f"{env_spec_path}/{SELF_REQUIREMENTS}"):
            cx.run(f"pip install -r {env_spec_path}/requirements.txt")

        else:
            print("No requirements.txt found")

        # if there is a 'self.requirements.txt' file specifying how to
        # install the package that is being worked on install it
        if osp.exists(f"{env_spec_path}/{SELF_REQUIREMENTS}"):
            cx.run(f"pip install -r {env_spec_path}/{SELF_REQUIREMENTS}")

        else:
            print("No self.requirements.txt found")

    print("----------------------------------------")
    print("to activate run:")
    print(f"source {venv_path}/bin/activate")

@task
def env(cx, name=DEFAULT_ENV):

    # choose your method:
    if ENV_METHOD == 'conda':
        conda_env(cx, name=name)

    elif ENV_METHOD == 'venv':
        venv_env(cx, name=name)

### Repo

@task(pre=[env,])
def repo_test(cx):

    # TODO: tests to run on the consistency and integrity of the repo

    # prefix all of these tests by activating the dev environment
    with cx.prefix(" fshank.dev"):
        cx.run("conda activate fshank.dev && inv -l",
               pty=True)

### Dependencies
# managing dependencies for the project at runtime

## pip: things that can be controlled by pip

@task
def deps_pip_pin(cx, name=DEFAULT_ENV):

    path = Path("envs") / name

    cx.run("pip-compile "
           f"--output-file={path}/requirements.txt "
           f"{path}/requirements.in")

    # SNIPPET: generate hashes is not working right, or just confusing me
    # cx.run("python -m piptools compile "
    #        "--generate-hashes "
    #        "--output-file=requirements.txt "
    #        f"requirements.in")

@task
def deps_pip_update(cx, name=DEFAULT_ENV):

    path = Path("envs") / name

    cx.run("python -m piptools compile "
           "--upgrade "
           f"--output-file={path}/requirements.txt "
           f"{path}/requirements.in")

## conda: managing conda dependencies

@task
def deps_conda_pin(cx, name=DEFAULT_ENV):

    env_spec_path = Path('envs') / name

    assert osp.exists(env_spec_path / 'env.yaml'), \
        "There must be an 'env.yaml' file to compile from"

    # make the environment under a mangled name so we don't screw with
    # the other one
    mangled_name = f"__mangled_{name}"
    env_dir = conda_env(cx, name=mangled_name)

    # pin to a 'env.pinned.yaml' file
    cx.run(f"conda env export "
           f"-p {env_dir} "
           f"-f {env_spec_path}/env.pinned.yaml")


    # then destroy the temporary mangled env
    cx.run(f"rm -rf {env_dir}")

@task
def deps_conda_update(cx, name=DEFAULT_ENV):

    # for now we just rewrite it
    deps_conda_pin(cx, name=name)

# altogether
@task
def deps_pin(cx, name=DEFAULT_ENV):

    deps_pip_pin(cx, name=name)
    deps_conda_pin(cx, name=name)

    # SNIPPET, IDEA: automatic git commits could be supported but
    # pairs poorly with the rest being automatic, would need better
    # semantics about splitting them up so they are analyzable, and no
    # such tool is planned so the consistency of commit messages is
    # unwarranted as of yet
    #
    # cx.run(f"git add -A && git commit -m 'pinned dependencies for the env: {name}'")



@task
def deps_pin_update(cx, name=DEFAULT_ENV):

    deps_pip_update(cx, name=name)

    deps_conda_update(cx, name=name)

    # SNIPPET, IDEA: automatic git commits could be supported but
    # pairs poorly with the rest being automatic, would need better
    # semantics about splitting them up so they are analyzable, and no
    # such tool is planned so the consistency of commit messages is
    # unwarranted as of yet
    #
    # cx.run(f"git add -A && git commit -m 'pinned dependencies for the env: {name}'")


### Cleaning

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
ORG_DOCS_SOURCES = [
    'changelog',
    'dev_guide',
    'general_info',
    'installation',
    'introduction',
    'news',
    'quick_start',
    'troubleshooting',
    'users_guide',
    'reference',
]

RST_DOCS_SOURCES = [
    'glossary',
    'tutorials/index',
]

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
        cx.run("sphinx-apidoc -f --separate --private --ext-autodoc --module-first --maxdepth 1 -o api ../src/fshank")

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
def tests_integration(cx, node=DEFAULT_ENV):
    cx.run(f"(cd tests/tests/test_integration && pytest -m 'not interactive')")

# @task
# def tests_unit(cx, node=DEFAULT_ENV):
#     cx.run(f"(cd tests/tests/test_unit && pytest -m 'not interactive' -m 'node_{node}')")

@task
def tests_unit(cx, node=DEFAULT_ENV):
    cx.run(f"(cd tests/tests/test_unit && pytest -m 'not interactive')")

@task
def tests_interactive(cx):
    """Run the interactive tests so we can play with things."""

    cx.run("pytest -m 'interactive'")

@task()
def tests_all(cx, node=DEFAULT_ENV):
    """Run all the automated tests. No benchmarks.

    There are different kinds of nodes that we can run on that
    different kinds of tests are available for.

    - minor : does not have a GPU, can still test most other code paths

    - dev : has at least 1 GPU, enough for small tests of all code paths

    - production : has multiple GPUs, good for running benchmarks
                   and full stress tests

    """

    cx.run(f"(cd tests/tests && pytest -m 'not interactive')")

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
    cx.run("flake8 --output-file=metrics/lint/flake8.txt src/fshank")

@task
def complexity(cx):
    """Analyze the complexity of the project."""

    cx.run("mkdir -p metrics/code_quality")

    cx.run("lizard -o metrics/code_quality/lizard.csv src/fshank")
    cx.run("lizard -o metrics/code_quality/lizard.html src/fshank")

    # SNIPPET: annoyingly opens the browser

    # make a cute word cloud of the things used
    # cx.run("(cd metrics/code_quality; lizard -EWordCount src/fshank > /dev/null)")

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


### Releases


## Pipeline

# Run code quality metrics

# Run Tests

# Run Performance Regressions

## version management

@task
def version_which(cx):
    """Tell me what version the project is at."""

    # get the current version
    from fshank import __version__ as proj_version
    print(proj_version)

# SNIPPET: not implemented yet

# @task
# def version_set(cx):
#     """Set the version with a custom string."""

#     print(NotImplemented)
#     NotImplemented


# # TODO: bumpversion is a flop don't use it. Just do a normal
# # replacement or do it manually
# @task
# def version_bump(cx, level='patch', new_version=None):
#     """Incrementally increase the version number by specifying the bumpversion level."""

#     print(NotImplemented)
#     NotImplemented

#     if new_version is None:
#         # use the bumpversion utility
#         cx.run(f"bumpversion --verbose "
#                 f"--new-version {new_version}"
#                 f"-m 'bumps version to {new_version}'"
#                 f"{level}")

#     elif level is not None:
#         # use the bumpversion utility
#         cx.run(f"bumpversion --verbose "
#                 f"-m 'bumps version level: {level}'"
#                 f"{level}")

#     else:
#         print("must either provide the level to bump or the version specifier")


#     # tag the git repo
#     cx.run("git tag -a ")


@task
def release_tag(cx, release=None):

    assert release is not None, "Release tag string must be given"

    tag_string = VCS_RELEASE_TAG_TEMPLATE.format(release)

    cx.run(f"git tag -a {tag_string} -m 'See the changelog for details'")


@task
def release(cx):

    # get the release version from the module
    from fshank import __version__ as proj_version

    # SNIPPET
    # cx.run("python -m wumpus.version")

    print("Releasing: ", proj_version)

    release_tag(cx, release=proj_version)


    # IDEA, TODO: handle the manual checklist of things


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

TESTING_INDEX_URL = "https://test.pypi.org/legacy/"

@task(pre=[clean_dist, build_sdist])
def publish_test_pypi(cx, version=None):

    assert version is not None

    cx.run("twine upload "
           f"--repository-url {TESTING_INDEX_URL} "
           "dist/*")

    # TODO: make it so that you don't have to throw out old dists and
    # just submit the ones we need for this version

    # cx.run("twine upload "
    #        f"--repository-url {TESTING_INDEX_URL} "
    #        "dist/fshank-{version}*")


@task(pre=[clean_dist, update_tools, build_sdist])
def publish_test(cx):

    from fshank import __version__ as version

    publish_test_pypi(cx, version=version)

# PyPI

PYPI_INDEX_URL = "https://upload.pypi.org/legacy/"

@task(pre=[clean_dist, build])
def publish_pypi(cx, version=None):

    assert version is not None

    cx.run(f"twine upload "
           f"--repository-url {PYPI_INDEX_URL} "
           f"dist/*")


@task
def publish_tags(cx, version=None):
    assert version is not None

    tag_string = VCS_RELEASE_TAG_TEMPLATE.format(version)

    cx.run(f"git push origin {tag_string}")

# TODO, SNIPPET, STUB: this is a desired target for uploading dists to github
# @task
# def publish_github_dists(cx, release=None):
#     assert release is not None, "Release tag string must be given"
#     pass

@task(pre=[clean_dist, update_tools, build])
def publish(cx, version=None):

    # assume the latest
    if version is None:
        from fshank import __version__ as version

    publish_tags(cx, version=version)
    publish_pypi(cx, version=version)

# Conda Forge

# TODO: convert to the regular conda forge repo when this is finished
# @task
# def conda_forge_recipe(cx):

#     print(NotImplemented)
#     NotImplemented

#     # TODO: somehow get this path right
#     CONDA_FORGE_RECIPE_PATH="../conda-forge_staged_recipe/recipes"
#     CONDA_FORGE_HASH_URL=""


#     # copy the recipe to the omnia fork
#     cx.run(f"cp conda/conda-forge {CONDA_FORGE_RECIPE_PATH}/fshank")

#     # commit and push
#     cx.run(f"git -C {CONDA_FORGE_RECIPE_PATH} commit -m 'update recipe'")
#     cx.run(f"git -C {CONDA_FORGE_RECIPE_PATH} push")
#     print(f"make a PR for this in the conda-forge conda recipes: {CONDA_FORGE_RECIPE_PATH}")

