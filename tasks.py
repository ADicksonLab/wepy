from invoke import task
import sys

PYTHON_VERSION='3.7'

# names of canned envs that might be useful here
TRIAL_ENV = 'wepy.trial'
DEV_ENV = 'wepy.dev'

CONDA_DEV_DEPS = ['pandoc', 'graphviz']
OMNIA_RUN_DEPS = ['openmm',]

# since we need to install openmmtools from pip since their conda
# install is a PITA
OPENMMTOOLS_DEPS = ['netcdf4', 'mpiplus', 'pymbar', 'pyyaml', 'cython',]
OPENMMTOOLS_INSTALL_TARGET = 'git+https://github.com/choderalab/openmmtools.git'

# OPENMM_CUDA_BUILD = "100"
# OPENMM_INSTALL_SUFFIX = f"-c omnia/label/cuda{OPENMM_CUDA_BUILD} openmm"


BENCHMARK_STORAGE_URL="./metrics/benchmarks"
BENCHMARK_STORAGE_URI="\"file://{}\"".format(BENCHMARK_STORAGE_URL)


### Repo

@task
def submodule_tests(ctx):
    """Retrieve the tests submodule if not already cloned."""

    ctx.run("git submodule update --init --recursive")
    ctx.run("git -C wepy-tests checkout master")


### Environments

@task
def deps_run(ctx):
    """Install bare minimum packages for running wepy with OpenMM."""


    ctx.run("pip install -r requirements.txt")

    ctx.run("conda install -y -c omnia {}".format(' '.join(OMNIA_RUN_DEPS)),
            pty=True)

@task(pre=[deps_run])
def deps_dev(ctx):
    """Install dependencies needed for developing in current env."""


    ctx.run("pip install -r dev.requirements.txt")

    ctx.run("conda install -y -c conda-forge {}".format(' '.join(CONDA_DEV_DEPS)),
            pty=True)


@task(pre=[deps_run])
def deps_examples(ctx):
    """Install dependencies needed for running the examples in current env."""


    # install the openmmtools dependencies
    ctx.run("conda install -y -c conda-forge {}".format(
        ' '.join(OPENMMTOOLS_DEPS)),
        pty=True)

    # then install openmmtools from pip
    ctx.run(f"pip install {OPENMMTOOLS_INSTALL_TARGET}",
            pty=True)


@task(pre=[deps_run, deps_dev, deps_examples])
def deps(ctx):
    """Install all dependencies for all auxiliary tasks in the repo in current env."""
    pass


@task
def env(cx, name='dev'):

    env_name=f"wepy.{name}"

    cx.run(f"conda create -y -n {env_name} python={PYTHON_VERSION}",
        pty=True)

    # install the pip dev dependencies
    cx.run(f"$ANACONDA_DIR/envs/{env_name}/bin/pip install -r envs/{name}.requirements.txt")

    # install the conda dev dependencies
    cx.run(f"conda env update -n {env_name} --file envs/{name}.env.yaml")

    # # install the pip tools dependencies
    # cx.run(f"$ANACONDA_DIR/envs/{env_name}/bin/pip install -r tools.requirements.txt")

    print("--------------------------------------------------------------------------------")
    print(f"run: conda activate {env_name}")



### Cleaning

@task
def clean_dist(ctx):
    """Remove all build products."""

    ctx.run("rm -rf dist build */*.egg-info *.egg-info")

@task
def clean_cache(ctx):
    """Remove all of the __pycache__ files in the packages."""
    ctx.run('find . -name "__pycache__" -exec rm -r {} +')

@task
def clean_docs(ctx):
    """Remove all documentation build products"""

    ctx.run("rm -rf sphinx/_build/*")

@task
def clean_website(ctx):
    """Remove all local website build products"""
    ctx.run("rm -rf docs/*")

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

        ctx.run(f"rm -rf {thing}")

@task(pre=[clean_cache, clean_dist, clean_docs, clean_website])
def clean(ctx):
    pass

### Building and packaging

@task
def sdist(ctx):
    """Make a source distribution"""
    ctx.run("python setup.py sdist")

### Docs

@task
def docs_build(ctx):
    """Buld the documenation"""
    ctx.run("(cd sphinx; ./build.sh)")

@task(pre=[docs_build])
def docs_serve(ctx):
    """Local server for documenation"""
    ctx.run("python -m http.server -d sphinx/_build/html")

### TODO: WIP Website

@task(pre=[clean_docs, clean_website, docs_build])
def website_deploy_local(ctx):
    """Deploy the docs locally for development. Must have bundler and jekyll installed"""

    # WIP: a more landing page style website for wepy using jekyll
    # which will have the docs linked to from it

    ctx.cd("jekyll")

    # update dependencies
    ctx.run("bundle install")
    ctx.run("bundle update")

    # run the server
    ctx.run("bundle exec jekyll serve")

@task(pre=[clean_docs, docs_build])
def website_deploy(ctx):
    """Deploy the documentation onto the internet."""

    ctx.run("(cd sphinx; ./deploy.sh)")



### Tests


@task
def tests_benchmarks(cx):
    cx.run("(cd wepy-tests/tests/test_benchmarks && pytest -m 'not interactive')")

@task
def tests_integration(cx, node='dev'):
    cx.run(f"(cd wepy-tests/tests/test_integration && pytest -m 'not interactive' -m 'node_{node}')")

@task
def tests_unit(cx, node='dev'):
    cx.run(f"(cd wepy-tests/tests/test_unit && pytest -m 'not interactive' -m 'node_{node}')")

@task
def tests_interactive(cx):
    """Run the interactive tests so we can play with things."""

    cx.run("pytest -m 'interactive'")

@task()
def tests_all(cx, node='dev'):
    """Run all the automated tests. No benchmarks.

    There are different kinds of nodes that we can run on that
    different kinds of tests are available for.

    - minor : does not have a GPU, can still test most other code paths

    - dev : has at least 1 GPU, enough for small tests of all code paths

    - production : has multiple GPUs, good for running benchmarks
                   and full stress tests

    """


    tests_unit(cx, node=node)
    tests_integration(cx, node=node)

@task
def tests_tox(ctx):

    NotImplemented

    TOX_PYTHON_DIR=None

    ctx.run("env PATH=\"{}/bin:$PATH\" tox".format(
        TOX_PYTHON_DIR))

### Code Quality

@task
def lint(ctx):

    ctx.run("flake8 src/wepy")

@task
def complexity(ctx):
    """Analyze the complexity of the project."""

    ctx.run("lizard -o metrics/code_quality/lizard.csv src/wepy")
    ctx.run("lizard -o metrics/code_quality/lizard.html src/wepy")

    # make a cute word cloud of the things used
    ctx.run("(cd metrics/code_quality; lizard -EWordCount src/wepy > /dev/null)")

### Profiling and Performance

@task
def profile(ctx):
    NotImplemented

@task
def benchmark_adhoc(ctx):
    """An ad hoc benchmark that will not be saved."""

    ctx.run("pytest wepy-tests/tests/test_benchmarks")

@task
def benchmark_save(ctx):
    """Run a proper benchmark that will be saved into the metrics for regression testing etc."""

    run_command = \
f"""pytest --benchmark-autosave --benchmark-save-data \
          --benchmark-storage={BENCHMARK_STORAGE_URI} \
          wepy-tests/tests/test_benchmarks
"""

    ctx.run(run_command)

@task
def benchmark_compare(ctx):

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

    ctx.run(run_command)


### Releases


## version management

@task
def version_which(ctx):
    """Tell me what version the project is at."""

    # get the current version
    import wepy
    print(wepy.__version__)


@task
def version_set(ctx):
    """Set the version with a custom string."""

    NotImplemented

@task
def version_bump(ctx, level='patch'):
    """Incrementally increase the version number by specifying the bumpversion level."""

    NotImplemented

    # use the bumpversion utility
    ctx.run("bumpversion {}".format(level))

    # tag the git repo
    ctx.run("git tag -a ")


## Publishing

@task(pre=[sdist])
def upload_pypi(ctx):
    ctx.run('twine upload dist/*')


## Scrapyard

# @task
# def env_dev(ctx):
#     """Recreate from scratch the wepy development environment."""

#     ctx.run(f"conda create -y -n {DEV_ENV} python={PYTHON_VERSION}",
#         pty=True)

#     # install wepy
#     ctx.run(f"$ANACONDA_DIR/envs/{DEV_ENV}/bin/pip install -e .")

#     # install the dev dependencies
#     ctx.run(f"$ANACONDA_DIR/envs/{DEV_ENV}/bin/pip install -r dev.requirements.txt")
#     ctx.run("conda install -y -c conda-forge {}".format(' '.join(CONDA_DEV_DEPS)),
#             pty=True)

#     # install the openmmtools dependencies
#     ctx.run("conda install -y -n {} -c conda-forge {}".format(
#         DEV_ENV,
#         ' '.join(OPENMMTOOLS_DEPS)),
#         pty=True)

#     # then install openmmtools from pip
#     ctx.run("$ANACONDA_DIR/envs/" + DEV_ENV + f"/bin/pip install {OPENMMTOOLS_INSTALL_TARGET}",
#             pty=True)

#     print("\n--------------------------------------------------------------")
#     print(f"Created an environment for trying out wepy called '{DEV_ENV}'")
#     print(f"enter it by running the command 'conda activate {DEV_ENV}'\n")



# @task
# def env_trial(ctx):
#     """Create (or recreate) an environment that is ready for running
#     pre-made wepy examples."""

#     ctx.run('echo "ANACONDA_DIR: $ANACONDA_DIR"')

#     ctx.run(f"conda create -y -n {TRIAL_ENV}",
#         pty=True)

#     ctx.run(f"conda install -y -n {TRIAL_ENV} python={PYTHON_VERSION}",
#         pty=True)

#     # install wepy
#     ctx.run("$ANACONDA_DIR/envs/" + TRIAL_ENV + "/bin/pip install -e .",
#             pty=True)

#     # install the other things needed for running the canned examples

#     # install the openmmtools dependencies
#     ctx.run("conda install -y -n {} -c conda-forge {}".format(
#         TRIAL_ENV,
#         ' '.join(OPENMMTOOLS_DEPS)),
#         pty=True)

#     # then install openmmtools from pip
#     ctx.run("$ANACONDA_DIR/envs/" + TRIAL_ENV + f"/bin/pip install {OPENMMTOOLS_INSTALL_TARGET}",
#             pty=True)

#     print("\n--------------------------------------------------------------")
#     print(f"Created an environment for trying out wepy called '{TRIAL_ENV}'")
#     print(f"enter it by running the command 'conda activate {TRIAL_ENV}'\n")
