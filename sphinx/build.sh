#!/bin/sh

# run pandoc to produce rst from org-mode files
for f in source/*.org; do
    echo "converting $f to rst"
    f_name=$(basename $f)
    pandoc -f org -t rst -o source/${f_name%.org}.rst $f
done;

# now build the API docs automatically from the source

# also build locally in sphinx for easier testing
make clean
# clean the api docs first
rm -rf api/*
# generate the api rst files for autodoc
sphinx-apidoc -f --separate --private --ext-autodoc --module-first --maxdepth 1 -o api ../src/wepy
sphinx-build -b html -E -a -j 6 . ./_build/html/
