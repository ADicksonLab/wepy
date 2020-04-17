from invoke import task

def tangle_orgfile(cx, file_path):
    """Tangle the target file using emacs in batch mode. Implicitly dumps
    things relative to the file."""

    cx.run(f"emacs -Q --batch -l org {file_path} -f org-babel-tangle")

@task
def init(cx):
    cx.run("mkdir -p _tangle_source")
    cx.run("mkdir -p _output")

@task
def clean(cx):
    cx.run("rm -rf _tangle_source")
    cx.run("rm -rf _output")

@task(pre=[init])
def tangle(cx):
    tangle_orgfile(cx, "README.org")
