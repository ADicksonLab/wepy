import nox

from pathlib import Path

@nox.session(
    python=['3.6', '3.7', '3.8', 'pypy3']
)
def test_user(session):
    """Test using basic pip based installation."""

    session.install("-r", ".jubeo/requirements.txt")
    session.install("-r", "envs/test/requirements.in")
    session.install("-r", "envs/test/self.requirements.txt")

    # TODO: add dates or commits or something to the tags

    session.run("inv", "py.tests-all", "-t", f"test-user_{session.python}")

@nox.session(
    python=['3.6', '3.7', '3.8'],
    venv_backend="conda",
)
def test_user_conda(session):
    """Test with conda as the installer."""

    # install the pip things first
    session.install("-r", ".jubeo/requirements.txt")
    session.install("-r", "envs/test/requirements.in")
    session.install("-r", "envs/test/self.requirements.txt")

    # install conda specific things here

    session.run("inv", "py.tests-all", "-t", f"test-user-conda_{session.python}")

@nox.session(
    python=['3.6', '3.7', '3.8', 'pypy3'],
)
def test_doc_pages(session):
    """Test using the env in each example directory."""

    session.install("-r", ".jubeo/requirements.txt")
    session.install("-r", "envs/test/requirements.in")
    session.install("-r", "envs/test/self.requirements.txt")

    # TODO replace with doit tasks to make things faster. this needs
    # to be done beforehand because the build env is particular, and
    # is different than the test env

    # session.run("inv", "docs.tangle")

    session.run("inv", "docs.test-pages", "-t", f"test-docs-pages_{session.python}")


@nox.session(
    python=['3.6', '3.7', '3.8'],
)
def test_example(session):
    """Test using the env in each example directory."""

    # grab the example from the command line
    assert len(session.posargs) == 1, \
        "Must provide exactly one example to run"

    example = session.posargs[0]

    # start installing things
    session.install("-r", ".jubeo/requirements.txt")

    # install the test dependencies
    test_requirements = Path(f"envs/test/requirements.txt")

    assert test_requirements.is_file()

    session.install("-r", str(test_requirements))

    requirements = Path(f"info/examples/{example}/env/requirements.txt")

    if requirements.is_file():
        session.install("-r", str(requirements))

    conda_env = Path(f"info/examples/{example}/env/env.pinned.yaml")

    if conda_env.is_file():

        session._run(*[
                'conda',
                'env',
                'update',
                '--prefix',
                session.virtualenv.location,
                '--file',
                str(conda_env),
                '--prune',
        ],
                     silent=True)

    session.run(*[
        "pytest",
        f"tests/test_docs/test_examples/test_{example}.py",
    ])
    # session.run("inv", "docs.test-example",
    #             "-n", f"{example}",
    #             "-t", f"test-docs_example-{example}_{session.python}")
