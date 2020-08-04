from invoke import task

# from ..config import ()

import os
import os.path as osp
from pathlib import Path
import shutil as sh
from warnings import warn

## Paths for the different things

DOCS_TEST_DIR = "tests/test_docs/_tangled_docs"
DOCS_EXAMPLES_DIR = "tests/test_docs/_examples"
DOCS_TUTORIALS_DIR = "tests/test_docs/_tutorials"

DOCS_SPEC = {
    'LANDING_PAGE' : "README.org",

    'INFO_INDEX' : "info/README.org",
    'QUICK_START' : "info/quick_start.org",
    'INTRODUCTION' : "info/introduction.org",
    'INSTALLATION' : "info/installation.org",
    'USERS_GUIDE' : "info/users_guide.org",
    'HOWTOS' : "info/howtos.org",
    'REFERENCE' : "info/reference.org",
    'TROUBLESHOOTING' : "info/troubleshooting.org",

    'GLOSSARY' : "info/glossary.rst",
    'BIBLIOGRAPHY' : "info/docs.bib",

    'DEV_GUIDE' : "info/dev_guide.org",
    'GENERAL' : "info/general_info.org",
    'NEWS' : "info/news.org",
    'CHANGELOG' : "info/changelog.org",

    'EXAMPLES_DIR' : "info/examples",
    'EXAMPLES_LISTING_INDEX' : "info/examples/README.org",

    # Other examples must be in a directory in the EXAMPLES_DIR and have
    # their own structure:

    # potentially literate document with source code. If not literate then
    # code should be in the EXAMPLE_SOURCE directory. This index should
    # still exist and give instructions on how to use and run etc. tangled
    # source will go in the EXAMPLE_TANGLE_SOURCE folder.
    'EXAMPLE_INDEX' : "README.org",

    'EXAMPLE_TASKS' : "tasks.py",
    'EXAMPLE_BUILD' : "dodo.py",

    # Source code for example that is not literately included in the
    # README.org
    'EXAMPLE_SOURCE' : "source",

    # included in the source tree
    'EXAMPLE_INPUT' : "input",

    # values are automatically excluded from the source tree via
    # .gitignore
    'EXAMPLE_OUTPUT' : "_output",

    # the directory that tangled source files will go, separate from the
    # source dir, this folder will be ignored by VCS
    'EXAMPLE_TANGLE_SOURCE' : "_tangle_source",

    # the actual dir the env will be built into
    'EXAMPLE_ENV' : "_env",

    'TUTORIALS_DIR' : "info/tutorials",
    'TUTORIALS_LISTING_INDEX' : "info/tutorials/README.org",

    # Other tutorials must be in a directory in the TUTORIALS_DIR and have
    # their own structure:

    # the main document for the tutorial can be *one* of any of the
    # values supporting: org, Jupyter Notebooks. In order of
    # precedence.
    'TUTORIAL_INDEX' : (
        "README.org",
        "README.ipynb",
    ),

    'TUTORIAL_TASKS' : "tasks.py",
    'TUTORIAl_BUILD' : "dodo.py",

    # Source code for tutorial that is not literately included in the
    # README.org
    'TUTORIAL_SOURCE' : "source",

    # included in the source tree
    'TUTORIAL_INPUT' : "input",

    # values are automatically excluded from the source tree via
    # .gitignore
    'TUTORIAL_OUTPUT' : "_output",

    # the directory that tangled source files will go, separate from the
    # source dir, this folder will be ignored by VCS
    'TUTORIAL_TANGLE_SOURCE' : "_tangle_source",

    # the actual dir the env will be built into
    'TUTORIAL_ENV' : "_env",

}

# here for reference potentially could be applied with an init function
GITIGNORE_LINES = [
    "info/examples/*/_output",
    "info/examples/*/_tangle_source",
    "info/examples/*/_env",
    "info/tutorials/*/_output",
    "info/tutorials/*/_tangle_source",
    "info/tutorials/*/_env",
]

# TODO: add a docs init task that generates all the files and adds to
# the gitignore.

def visit_docs():
    """Returns a list of all the doc pages with their relative paths to
    the root of the project. Not including examples and tutorials
    which are tested differently.

    """

    # get the pages which are always there
    page_keys = [
        'LANDING_PAGE',
        'INFO_INDEX',
        'QUICK_START',
        'INTRODUCTION',
        'INSTALLATION',
        'USERS_GUIDE',
        'HOWTOS',
        'REFERENCE',
        'TROUBLESHOOTING',
        'GLOSSARY',
        'DEV_GUIDE',
        'GENERAL',
        'NEWS',
        'CHANGELOG',
        'EXAMPLES_LISTING_INDEX',
        'TUTORIALS_LISTING_INDEX',
    ]

    # dereference their paths
    page_paths = [DOCS_SPEC[key] for key in page_keys]

    return page_paths

def visit_examples():
    """Get the relative paths to all of the example dirs."""

    # get the pages for the tutorials and examples
    examples = [ex for ex in os.listdir(DOCS_SPEC['EXAMPLES_DIR'])
                if (
                        ex != Path(DOCS_SPEC['EXAMPLES_LISTING_INDEX']).parts[-1] and
                        ex != '.keep' and
                        not ex.endswith("~")
                )
    ]

    example_dirs = [Path(DOCS_SPEC['EXAMPLES_DIR']) / example for example in examples]

    return example_dirs

def visit_example_contents(example):

        example_pages = []
        if osp.exists(DOCS_SPEC['EXAMPLE_INDEX']):
            example_index = example_dir / DOCS_SPEC['EXAMPLE_INDEX']
            example_pages.append(example_index)
        else:
            warn(f"No example index page for {example}")

        page_paths.extend(example_pages)

def visit_tutorials():
    """Get the relative paths to all of the tutorial dirs."""

    # get the pages for the tutorials and tutorials
    tutorials = [tut for tut in os.listdir(DOCS_SPEC['TUTORIALS_DIR'])
                if (
                        tut != Path(DOCS_SPEC['TUTORIALS_LISTING_INDEX']).parts[-1] and
                        tut != 'index.rst' and
                        tut != '.keep' and
                        not tut.endswith("~")
                )
    ]

    tutorial_dirs = [Path(DOCS_SPEC['TUTORIALS_DIR']) / tutorial for tutorial in tutorials]

    return tutorial_dirs

def tangle_orgfile(cx, file_path):
    """Tangle the target file using emacs in batch mode. Implicitly dumps
    things relative to the file."""

    cx.run(f"emacs -Q --batch -l org {file_path} -f org-babel-tangle")

def tangle_jupyter(cx, file_path):
    """Tangle the target file using jupyter-nbconvert to a python
    script. Implicitly dumps things relative to the file. Only can
    make a single script from the notebook with the same name.

    """

    cx.run(f"jupyter-nbconvert --to 'python' {file_path}")


@task
def list_docs(cx):
    """List paths relative to this context"""

    print('\n'.join([str(Path(cx.cwd) / p) for p in visit_docs()]))

@task
def list_examples(cx):
    """List paths relative to this context"""

    print('\n'.join([str(Path(cx.cwd) / ex) for ex in visit_examples()]))

@task
def list_tutorials(cx):
    """List paths relative to this context"""

    print('\n'.join([str(Path(cx.cwd) / tut) for tut in visit_tutorials()]))

@task()
def clean_tangle(cx):
    """remove the tangle dirs"""

    sh.rmtree(Path(cx.cwd) / DOCS_TEST_DIR,
              ignore_errors=True)

    sh.rmtree(Path(cx.cwd) / DOCS_EXAMPLES_DIR,
              ignore_errors=True)

    sh.rmtree(Path(cx.cwd) / DOCS_TUTORIALS_DIR,
              ignore_errors=True)


@task(pre=[clean_tangle])
def tangle_pages(cx):
    """Tangle the docs into the docs testing directory."""

    docs_test_dir = Path(cx.cwd) / DOCS_TEST_DIR

    os.makedirs(
        docs_test_dir,
        exist_ok=True,
    )

    doc_pages = visit_docs()
    for page_path in doc_pages:

        page_path = Path(page_path)

        page_name_parts = page_path.parts[0:-1] + (page_path.stem,)
        page_name = Path(*page_name_parts)
        page_type = page_path.suffix.strip('.')

        page_tangle_dir = docs_test_dir / page_name
        # make a directory for this file to have it's own tangle environment
        os.makedirs(page_tangle_dir,
                    exist_ok=False)

        # copy the page to its directory
        target_orgfile = docs_test_dir / page_name / f"{page_name.stem}.{page_type}"
        sh.copyfile(page_path,
                    target_orgfile)

        # then tangle them
        tangle_orgfile(cx, target_orgfile)

@task(pre=[clean_tangle])
def tangle_examples(cx):

    examples_test_dir = Path(cx.cwd) / DOCS_EXAMPLES_DIR

    os.makedirs(
        examples_test_dir,
        exist_ok=True,
    )

    for example_dir in visit_examples():

        example = example_dir.stem

        # ignore if there are any built files at the start location,
        # need to build fresh for tests
        sh.copytree(
            example_dir,
            examples_test_dir / example,
            ignore=sh.ignore_patterns("_*"),
        )

        with cx.cd(str(examples_test_dir / example)):
            cx.run("inv clean")
            cx.run("inv tangle")


@task(pre=[clean_tangle])
def tangle_tutorials(cx):

    tutorials_test_dir = Path(cx.cwd) / DOCS_TUTORIALS_DIR

    os.makedirs(
        tutorials_test_dir,
        exist_ok=True,
    )

    for tutorial_dir in visit_tutorials():

        tutorial = tutorial_dir.stem

        # ignore if there are any built files at the start location,
        # need to build fresh for tests
        sh.copytree(
            tutorial_dir,
            tutorials_test_dir / tutorial,
            ignore=sh.ignore_patterns("_*"),
        )


        with cx.cd(str(tutorials_test_dir / tutorial)):
            cx.run("inv clean")
            cx.run("inv tangle")


@task(pre=[clean_tangle, tangle_pages, tangle_examples, tangle_tutorials])
def tangle(cx):
    """Tangle the doc pages, examples, and tutorials into the docs testing
    directories."""

    pass


@task
def new_example(cx, name=None, template="org", env='venv_blank'):
    """Create a new example in the info/examples directory.

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

    template_path = Path(f"templates/examples/{template}")

    # check if the template exists
    if not template_path.is_dir():

        raise ValueError(
            f"Unkown template {template}. Check the 'templates/examples' folder")

    # check if the env exists
    if env is not None:
        env_tmpl_path = Path(f"templates/envs/{env}")

        if not env_tmpl_path.is_dir():

            raise ValueError(
                f"Unkown env template {env}. Check the 'templates/envs' folder")

    target_path = Path(f"info/examples/{name}")

    if target_path.exists():
        raise FileExistsError(f"Example with name {name} already exists. Not overwriting.")

    # copy the template
    cx.run(f"cp -r {template_path} {target_path}")

    # copy the env
    cx.run(f"cp -r {env_tmpl_path} {target_path / 'env'}")

    print(f"New example created at: {target_path}")

@task
def new_tutorial(cx, name=None, template="org", env='venv_blank'):
    """Create a new tutorial in the info/tutorials directory.

    Can choose between the following templates:
    - 'org' :: org mode notebook
    - 'jupyter' :: Jupyter notebook


    Choose from the following env templates:

    - None
    - venv_blank
    - venv_dev
    - conda_blank
    - conda_dev

    """

    assert name is not None, "Must provide a name"

    template_path = Path(f"templates/tutorials/{template}")

    # check if the template exists
    if not template_path.is_dir():

        raise ValueError(
            f"Unkown template {template}. Check the 'templates/tutorials' folder")

    # check if the env exists
    if env is not None:
        env_tmpl_path = Path(f"templates/envs/{env}")

        if not env_tmpl_path.is_dir():

            raise ValueError(
                f"Unkown env template {env}. Check the 'templates/envs' folder")


    target_path = Path(f"info/tutorials/{name}")

    if target_path.exists():
        raise FileExistsError(f"Tutorial with name {name} already exists. Not overwriting.")

    # copy the template
    cx.run(f"cp -r {template_path} {target_path}")

    # copy the env
    cx.run(f"cp -r {env_tmpl_path} {target_path / 'env'}")

    print(f"New tutorial created at: {target_path}")


@task
def test_example(cx,
                 name=None,
                 tag=None,
):
    """Test a specific doc example in the current virtual environment."""

    if name is None:
        examples = visit_examples()
    else:
        examples = [Path("info/examples") / name]

    for example in examples:

        path = example

        assert path.exists() and path.is_dir(), \
            f"Example {example.stem} doesn't exist at {path}"

        # TODO: add support for reports and such
        print("tag is ignored")

        cx.run(f"pytest tests/test_docs/test_examples/test_{example.stem}.py",
               warn=True)

@task
def test_examples_nox(cx,
                  name=None):
    """Test either a specific example when 'name' is given or all of them,
    using the nox test matrix specified in the noxfile.py file for
    'test_example' session.

    """

    if name is None:
        examples =  [example.stem for example in visit_examples()]
    else:
        examples = [name]

    for example in examples:
        cx.run(f"nox -s test_example -- {example}",
               warn=True)

@task
def test_tutorial(cx,
                 name=None,
                 tag=None,
):
    """Test a specific doc tutorial in the current virtual environment."""


    if name is None:
        tutorials = visit_tutorials()
    else:
        tutorials = [Path("info/tutorials") / name]

    for tutorial in tutorials:

        path = tutorial

        assert path.exists() and path.is_dir(), \
            f"Tutorial {tutorial} doesn't exist at {path}"

        # TODO: add support for reports and such
        print("tag is ignored")

        cx.run(f"pytest tests/test_docs/test_tutorials/test_{tutorial.stem}.py",
               warn=True)

@task
def test_tutorials_nox(cx,
                  name=None):
    """Test either a specific tutorial when 'name' is given or all of them,
    using the nox test matrix specified in the noxfile.py file for
    'test_tutorial' session.

    """

    if name is None:
        tutorials =  [tutorial.stem for tutorial in visit_tutorials()]
    else:
        tutorials = [name]

    for tutorial in tutorials:
        cx.run(f"nox -s test_tutorial -- {tutorial}")


@task
def test_pages(cx, tag=None):
    """Test the doc pages in the current virtual environment."""

    if tag is None:
        cx.run("pytest tests/test_docs/test_pages",
               warn=True)

    else:
        cx.run(f"pytest --html=reports/pytest/{tag}/docs/report.html tests/test_docs/test_pages",
               warn=True)


@task
def test_pages_nox(cx, tag=None):
    """Test the doc pages in the nox test matrix session."""

    cx.run(f"nox -s test_doc_pages")

@task
def pin_example(cx, name=None):
    """Pin the deps for an example or all of them if 'name' is None."""

    if name is None:
        examples = visit_examples()
    else:
        examples = [name]

    print(examples)
    for example in examples:

        path = example / 'env'

        assert path.exists() and path.is_dir(), \
            f"Env for Example {example} doesn't exist"

        cx.run(f"inv env.deps-pin-path -p {path}")

@task
def pin_tutorial(cx, name=None):

    if name is None:
        tutorials = visit_tutorials()
    else:
        tutorials = [name]

    for tutorial in tutorials:

        path = tutorial / 'env'

        assert path.exists() and path.is_dir(), \
            f"Env for Tutorial {tutorial} doesn't exist"

        cx.run(f"inv env.deps-pin-path -p {path}")

@task
def env_example(cx, name=None):
    """Make a the example env in its local dir."""

    if name is None:
        examples = visit_examples()
    else:
        examples = [name]

    for example in examples:


        spec_path = Path(DOCS_SPEC['EXAMPLES_DIR']) / example / 'env'
        env_path = Path(DOCS_SPEC['EXAMPLES_DIR']) / example / DOCS_SPEC['EXAMPLE_ENV']

        assert spec_path.exists() and spec_path.is_dir(), \
            f"Tutorial {example} doesn't exist"

        cx.run(f"inv env.make-env -s {spec_path} -p {env_path}")

@task
def env_tutorial(cx, name=None):
    """Make a the tutorial env in its local dir."""

    if name is None:
        tutorials = visit_tutorials()
    else:
        tutorials = [name]

    for tutorial in tutorials:


        spec_path = Path(DOCS_SPEC['TUTORIALS_DIR']) / tutorial / 'env'
        env_path = Path(DOCS_SPEC['TUTORIALS_DIR']) / tutorial / DOCS_SPEC['TUTORIAL_ENV']

        assert spec_path.exists() and spec_path.is_dir(), \
            f"Tutorial {tutorial} doesn't exist"

        cx.run(f"inv env.make-env -s {spec_path} -p {env_path}")

