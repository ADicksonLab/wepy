# Contributing 

Developing on wepy

After you have cloned the repo you should have a copy of Python 3.10 installed
and accessible to path as `python3.10`.

You will also need to install [nox](https://nox.thea.codes/) on your system. I
recommend `pipx`.

Then you should be able to start coding and running tests:

```sh
make test-unit
```

To build the package:

```sh
make build
```

To see all the commands run `make` or:

```sh
make help
```

All the virtual environment creation and installation is taken care behind the
scenes.

If you want a standalone environment you can activate in your shell run:

```sh
make env
```

Which creates the `.venv` folder. For convenience it can be activated with:

```sh
. ./env.sh
```


## TODOs

Note that there are stubs for performing checks like linting, formatting etc.
that aren't used in this project, but could be added at a later time.

In order to get tests passing many bad tests were deleted.

Building of documentation is also not implemented at the moment.
