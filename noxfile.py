# NOTE: A quick note on what Nox is used for specifically. Nox is used for
# anything that requires some sort of special virtual environment in order to
# operate. That includes creating standalone virtualenvs, and for single tasks
# requiring a specific environment. Nox should be considered an implementation
# detail of this however and all relevant high level targets should be still
# created in the Makefile. Furthermore, git hooks should reference those
# Makefile targets rather than the nox targets directly; keeping them decoupled.
# Of course feel free to use the nox targets if its easier.

# Standard Library
import itertools as it
import os
from pathlib import Path

# Third Party Library
import nox

# exclude the 'dev' session here so its not run automatically
nox.options.sessions = []

# NOTE: that with 3.11 mdtraj fails to build
DEFAULT_PYTHON_VERSION = "3.10"

PROJECT_ROOT_DIR = Path(__file__).parent

SRC_DIR = PROJECT_ROOT_DIR / "src"

SPHINX_SOURCE_DIR = PROJECT_ROOT_DIR / "sphinx"
SPHINX_BUILD_DIR = SPHINX_SOURCE_DIR / "_build"

# listing of things to be formatted and checked
FORMAT_TARGETS = [
    "src",
    "tests",
    "noxfile.py",
    SPHINX_SOURCE_DIR / "conf.py",
]

LINT_TARGETS = FORMAT_TARGETS

TYPECHECK_TARGETS = []

UNIT_TEST_DIRNAME = "unit"

### Helpers


def install_requirements(requirements_paths: list[str]) -> list[str]:
    """Given a list of requirements files generate the subprocess string for
    installing all of them."""

    return list(
        it.chain(
            *it.zip_longest(
                [],
                requirements_paths,
                fillvalue="-r",
            )
        )
    )


def install_interactive(session: nox.Session) -> None:
    """Install the standard set of interactive work dependencies, not useful in CI typically."""

    session.install("-r", "dev/interactive.requirements.txt")


### Pinning

# which extras to generate pin files for
PIN_EXTRAS = [
    "md",
    "distributed",
    "prometheus",
    "graphics",
]

EXTRAS_REQUIREMENTS_MAP = {
    "md": "requirements-md.txt",
    "distributed": "requirements-distributed.txt",
    "prometheus": "requirements-prometheus.txt",
}


def resolve_extras_reqfiles(extras: str) -> list[str]:
    extras_items = extras.split(",")

    extras_reqfiles = []
    for extra in extras_items:
        extras_reqfiles.append(EXTRAS_REQUIREMENTS_MAP[extra])

    return extras_reqfiles


@nox.session
@nox.parametrize("extras", PIN_EXTRAS)
def pin(session, extras):
    session.install("pip-deepfreeze")
    session.run("pip-df", "sync", "--extras", ",".join(PIN_EXTRAS))


### Development Environment

# this VENV_DIR constant specifies the name of the dir that the `dev`
# session will create, containing the virtualenv;
# the `resolve()` makes it portable
DEV_VENV_DIR = Path("./.venv").resolve()


def external_venv(
    session,
    requirements_txt_list: list[str],
    venv_path: Path = DEV_VENV_DIR,
):
    session.install("virtualenv")
    session.run("virtualenv", os.fsdecode(venv_path), silent=True)

    python = os.fsdecode(venv_path.joinpath("bin/python"))

    install_spec = install_requirements(requirements_txt_list)

    session.run(
        python,
        "-m",
        "pip",
        "install",
        *install_spec,
        "-e",
        ".",
        external=True,
    )


@nox.session(python=DEFAULT_PYTHON_VERSION)
def dev_external(session: nox.Session) -> None:
    """Set up a development environment in the '.venv' top-level folder.

    This development environment contains all dependencies needed for
    development.

    """

    mandatory_reqs = [
        "requirements.txt",
        "dev/qa.requirements.txt",
        "dev/typechecking.requirements.txt",
        "dev/testing.requirements.txt",
    ]

    # we use all the extras for the dev environment
    extras_reqs = list(EXTRAS_REQUIREMENTS_MAP.values())

    base_reqs = mandatory_reqs + extras_reqs

    if session.interactive:
        reqs = base_reqs + ["dev/interactive.requirements.txt"]

    else:
        reqs = base_reqs

    external_venv(
        session,
        reqs,
    )


@nox.session(python=DEFAULT_PYTHON_VERSION)
@nox.parametrize(
    "extras",
    [
        "postgres",
        "postgres-async",
        "postgres,postgres-async",
    ],
)
def prod_external(
    session: nox.Session,
    extras: str,
) -> None:
    extra_reqs = [f"requirements-{extra}.txt" for extra in extras.split(",")]

    external_venv(
        session,
        ["requirements.txt"] + extra_reqs,
    )


### QA

## Base Functions


def _black_format(session):
    session.run("black", *FORMAT_TARGETS)


def _isort_format(session):
    session.run("isort", *FORMAT_TARGETS)


def _format(session):
    _black_format(session)
    _isort_format(session)


def _black_check(session):
    session.run("black", "--check", *FORMAT_TARGETS)


def _isort_check(session):
    session.run("isort", "--check", *FORMAT_TARGETS)


def _format_check(session):
    _black_check(session)
    _isort_check(session)


def _flake8(session):
    session.run("flake8", *LINT_TARGETS)


def _interrogate(session):
    session.run("interrogate", *LINT_TARGETS)


def _docstring_lint(session):
    _interrogate(session)


def _lint(session):
    _flake8(session)


def _typecheck(session):
    session.run("mypy", "--strict", *TYPECHECK_TARGETS)


def qa_install(session):
    install_spec = install_requirements(
        [
            "dev/qa.requirements.txt",
        ]
    )

    session.install(*install_spec)


## Fine Grained Sessions
@nox.session
def black_check(session):
    qa_install(session)
    _black_check(session)


@nox.session
def isort_check(session):
    qa_install(session)
    _isort_check(session)


@nox.session
def format_check(session):
    qa_install(session)
    _format_check(session)


@nox.session
def flake8(session):
    qa_install(session)
    _flake8(session)


@nox.session
def interrogate(session):
    qa_install(session)
    _interrogate(session)


@nox.session
def docstring_lint(session):
    qa_install(session)
    _docstring_lint(session)


@nox.session
def lint(session):
    qa_install(session)
    _lint(session)


## Top-Level Targets
@nox.session
def validate(session: nox.Session) -> None:
    """Run all static analysis QA checks."""
    qa_install(session)

    _format_check(session)
    _lint(session)


@nox.session(python=DEFAULT_PYTHON_VERSION)
def typecheck(session):
    """Run typechecking for the project."""

    install_spec = install_requirements(
        [
            "dev/typechecking.requirements.txt",
        ]
    )

    session.install(*install_spec, "-e", ".")
    _typecheck(session)


@nox.session
def format(session):
    """Run formatting on the code."""
    qa_install(session)

    _format(session)


### Tests


@nox.session(python=DEFAULT_PYTHON_VERSION)
def tests_unit(
    session: nox.Session,
) -> None:
    """Run the unit tests, generate the coverage database and the HTML report."""

    install_spec = install_requirements(
        [
            "dev/testing.requirements.txt",
            "requirements.txt",
            # we add all the extras in as well, we don't use them
            # inappropriately though!
            "requirements-distributed.txt",
            "requirements-md.txt",
            "requirements-prometheus.txt",
        ]
    )

    session.install(*install_spec, "-e", ".")

    if session.interactive:
        install_interactive(session)

    session.run(
        "pytest",
        "-s",
        # modern way of importing stuff, use with `pythonpath` option in pytest.ini
        "--import-mode=importlib",
        "--cov-report=term-missing:skip-covered",
        "--cov=wepy",
        # for this stage don't fail on missing coverage
        "--cov-fail-under=0",
        # the pointer plugin, collect covered modules
        # "--pointers-collect=src",
        # "--pointers-report",
        # "--pointers-func-min-pass=1",
        # "--pointers-fail-under=100",
        # # block the loading of the integration test plugins
        # "-p",
        # "no:local_test_utils.plugins.database",
        # f"tests/{UNIT_TEST_DIRNAME}",
        "tests/unit/test_work_mapper",
    )

    session.run("coverage", "html", "--fail-under=100", "--skip-covered")


@nox.session(python=DEFAULT_PYTHON_VERSION)
def coverage(session):
    """Check that the coverage generated by unit tests passes."""

    # TODO: add a way to report the unit coverage as well

    session.install("coverage")

    session.run("coverage", "html", "--skip-covered")
    session.run(
        "coverage",
        "report",
        "--fail-under=100",
        "--data-file=.coverage",
        "--show-missing",
    )


# TODO: integration, benchmark, acceptance, and docs tests
# TODO: build documentation

## Builds & Releases


@nox.session(python=DEFAULT_PYTHON_VERSION)
def build(session):
    session.install("hatch")

    session.run("hatch", "build")


@nox.session(python=DEFAULT_PYTHON_VERSION)
def bumpversion(session):
    session.install("hatch")

    if session.posargs:
        assert len(session.posargs) == 1, "Too many arguments only need 1."
        part = session.posargs[0]
        assert part in (
            "major",
            "minor",
            "patch",
        ), "Must choose a valid bump part ('major', 'minor', 'patch')"
    else:
        part = "patch"

    session.run("hatch", "version", part)


# NOTE: this is the current owner of the PyPI package contact my email above for
# more info
PYPI_USER = "salotz"


@nox.session(python=DEFAULT_PYTHON_VERSION)
def publish(session):
    session.install("hatch")

    session.run(
        "hatch",
        "-v",
        "publish",
        env={
            "HATCH_INDEX_USER": PYPI_USER,
        },
    )
