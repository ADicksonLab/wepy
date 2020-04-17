from invoke import task

@task
def init(cx):
    cx.run("mkdir -p _output")
    cx.run("mkdir -p _tangle_source")

@task
def clean(cx):
    cx.run("rm -rf _output/*")
    cx.run("rm -rf _tangle_source/*")

@task(pre=[init])
def tangle(cx):
    cx.run("jupyter-nbconvert --to 'python' --output-dir=_tangle_source README.ipynb")
