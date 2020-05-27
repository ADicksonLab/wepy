from invoke import task

import os
import os.path as osp
from pathlib import Path

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
    cx.run(f"chmod ug+x ./_tangle_source/*.bash", warn=True)
    cx.run(f"chmod ug+x ./_tangle_source/*.sh", warn=True)
    cx.run(f"chmod ug+x ./_tangle_source/*.py", warn=True)

@task
def clean_env(cx):
    cx.run("rm -rf _env")

@task(pre=[init])
def env(cx):
    """Create the environment from the specs in 'env'. Must have the
    entire repository available as it uses the tooling from it.

    """

    example_name = Path(os.getcwd()).stem

    with cx.cd("../../"):
        cx.run(f"inv py.env-jig -n {example_name}")


### Custom tasks


## Prometheus

PROM_VERSION = '2.18.1'
PROM_FNAME = f"prometheus-{PROM_VERSION}.linux-amd64"
PROM_URL = f"https://github.com/prometheus/prometheus/releases/download/v{PROM_VERSION}/{PROM_FNAME}.tar.gz"

PROM_PORT = "9090"
@task
def prometheus_download(cx):

    cx.run("mkdir -p _output/programs")

    with cx.cd("_output/programs"):
        cx.run(f"wget {PROM_URL}", pty=True)
        cx.run(f"tar --extract -f {PROM_FNAME}.tar.gz")

@task(pre=[prometheus_download])
def prometheus_install(cx):

    with cx.cd("_output/programs"):
        cx.run(f"ln -s -r -f -T {PROM_FNAME}/prometheus prometheus")

@task
def grafana_url(cx):
    print(f"URL: http://localhost:{PROM_PORT}")

@task
def prometheus_start(cx):

    cx.run("mkdir -p _output/prometheus")

    grafana_url(cx)

    cx.run("./_output/programs/prometheus "
           "--config.file=./input/prometheus.yaml "
           "--storage.tsdb.path=./_output/prometheus/data"
    )

## grafana

GRAF_VERSION = '6.7.3'
GRAF_FNAME = f"grafana-{GRAF_VERSION}"
GRAF_URL = f"https://dl.grafana.com/oss/release/{GRAF_FNAME}.linux-amd64.tar.gz"
GRAF_PORT = "3000"

@task
def grafana_download(cx):

    cx.run("mkdir -p _output/programs")

    with cx.cd("_output/programs"):
        cx.run(f"wget {GRAF_URL}", pty=True)
        cx.run(f"tar --extract -f {GRAF_FNAME}.linux-amd64.tar.gz")

@task(pre=[grafana_download])
def grafana_install(cx):

    with cx.cd("_output/programs"):
        cx.run(f"ln -s -r -f -T {GRAF_FNAME} grafana")

@task
def grafana_url(cx):

    print(f"URL: http://localhost:{GRAF_PORT}")
    print("username: admin")
    print("password: admin")

@task
def grafana_start(cx):

    cx.run("mkdir -p _output/grafana")

    grafana_url(cx)


    cx.run("./_output/programs/grafana/bin/grafana-server "
           "--homepath ./_output/programs/grafana "
    )


## All services

@task
def service_urls(cx):

    prometheus_url(cx)
    grafana_url(cx)
