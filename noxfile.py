import nox

from pathlib import Path

# @nox.session(
#     python=['3.6', '3.7', '3.8', 'pypy3'],
# )
# def test_user(session):
#     """Test using basic pip based installation.

#     This won't be able to run things that are based on OpenMM as it
#     needs to be installed with conda.

#     """

#     session.install("-r", ".jubeo/requirements.txt")
#     session.install("-r", "envs/test/requirements.in")
#     session.install("-r", "envs/test/self.requirements.txt")

#     session.run("inv", "py.tests-all", "-t", f"test-user_{session.python}")

#    python=['3.6', '3.7', '3.8'],
@nox.session(
    python=['3.7',],
    venv_backend="conda",
)
def test_unit(session):
    """Test with conda as the installer.

    This is the full suite of tests and should be favored over the non
    conda one, because we need to test using OpenMM for a lot of
    things.

    """

    ## install the conda dependencies
    conda_env = Path(f"envs/test/env.pinned.yaml")

    print("Session location: ", session.virtualenv.location)
    if conda_env.is_file():

        session.run(
            'conda',
            'env',
            'update',
            '--prefix',
            session.virtualenv.location,
            '--file',
            str(conda_env),
            # options
            silent=True)

    # install the pip things first
    session.install("-r", ".jubeo/requirements.txt")
    session.install("-r", "envs/test/requirements.in")
    session.install("-r", "envs/test/self.requirements.txt")

    # install conda specific things here

    session.run("inv", "py.tests-unit", "-t", f"test-unit-conda_{session.python}")

@nox.session(
    python=['3.6', '3.7', '3.8'],
    venv_backend="conda",
)
def test_doc_pages(session):
    """Test using the env in each example directory.

    Tests using conda install with OpenMM dependency.

    """

    ## install the conda dependencies
    conda_env = Path(f"envs/test/env.pinned.yaml")

    print("Session location: ", session.virtualenv.location)
    if conda_env.is_file():

        session.run(
            'conda',
            'env',
            'update',
            '--prefix',
            session.virtualenv.location,
            '--file',
            str(conda_env),
            # options
            silent=True)

    session.install("-r", ".jubeo/requirements.txt")
    session.install("-r", "envs/test/requirements.in")
    session.install("-r", "envs/test/self.requirements.txt")

    session.run("inv", "docs.test-pages", "-t", f"test-docs-pages_{session.python}")


# DEBUG
#    python=['3.6', '3.7', '3.8'],
@nox.session(
    python=['3.7',],
    venv_backend="conda",
)
def test_example(session):
    """Test using the env in each example directory."""

    # grab the example from the command line
    assert len(session.posargs) == 1, \
        "Must provide exactly one example to run"

    example = session.posargs[0]

    ## install the conda deps,

    # NOTE: this must come first, before pip installing things

    conda_env = Path(f"info/examples/{example}/env/env.pinned.yaml")

    print("Session location: ", session.virtualenv.location)
    if conda_env.is_file():

        session.run(
            'conda',
            'env',
            'update',
            '--prefix',
            session.virtualenv.location,
            '--file',
            str(conda_env),
            # options
            silent=True)


    ## Install pip dependencies
    session.install("-r", ".jubeo/requirements.txt")

    # install the test dependencies
    test_requirements = Path(f"envs/test/requirements.txt")

    assert test_requirements.is_file()

    session.install("-r", str(test_requirements))

    # install the example requirements
    requirements = Path(f"info/examples/{example}/env/requirements.txt")
    if requirements.is_file():
        session.install("-r", str(requirements))

    # install the package under test
    self_requirements = Path(f"info/examples/{example}/env/self.requirements.txt")
    if self_requirements.is_file():
        session.install("-r", str(self_requirements))

    session.run(
        "pytest",
        f"tests/test_docs/test_examples/test_{example}.py",
    )

@nox.session(
    python=['3.6', '3.7', '3.8'],
    venv_backend="conda",
)
def test_tutorial(session):
    """Test using the env in each tutorial directory."""

    # grab the tutorial from the command line
    assert len(session.posargs) == 1, \
        "Must provide exactly one tutorial to run"

    tutorial = session.posargs[0]

    ## install the conda deps,

    # NOTE: this must come first, before pip installing things

    conda_env = Path(f"info/tutorials/{tutorial}/env/env.pinned.yaml")

    print("Session location: ", session.virtualenv.location)
    if conda_env.is_file():

        session.run(
            'conda',
            'env',
            'update',
            '--prefix',
            session.virtualenv.location,
            '--file',
            str(conda_env),
            # options
            silent=True)


    ## Install pip dependencies
    session.install("-r", ".jubeo/requirements.txt")

    # install the test dependencies
    test_requirements = Path(f"envs/test/requirements.txt")

    assert test_requirements.is_file()

    session.install("-r", str(test_requirements))

    requirements = Path(f"info/tutorials/{tutorial}/env/requirements.txt")

    if requirements.is_file():
        session.install("-r", str(requirements))


    session.run(
        "pytest",
        f"tests/test_docs/test_tutorials/test_{tutorial}.py",
    )
