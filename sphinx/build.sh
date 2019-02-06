# run pandoc on all the handwritted documentation files to produce rst
# from org files
rm source/*~* &> /dev/null
for f in $(ls source); do
    echo "converting source/$f to rst"
    pandoc -f org -t rst -o source/${f%.org}.rst source/$f
done;
rm source/*.rst.rst* &> /dev/null

# now build the API docs automatically from the source

# also build locally in sphinx for easier testing
make clean
# clean the api docs first
rm -rf api/*
# generate the api rst files for autodoc
sphinx-apidoc -f --separate --module-first --maxdepth 1 -o api ../wepy
sphinx-build -b html -E -a -j 6 . ./_build/html/
