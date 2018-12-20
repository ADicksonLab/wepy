# for building to the docs for github pages
# mkdir -p ../docs
# rm -rf ../docs/

# run pandoc on all the handwritted documentation files to produce rst
# from org files
rm source/*~* &> /dev/null
for f in $(ls source); do
    echo "converting source/$f to rst"
    pandoc -f org -t rst -o source/${f%.org}.rst source/$f
done;
rm source/*.rst.rst* &> /dev/null

# now build the API docs automatically from the source

# # build to the docs dir
# sphinx-apidoc -f -o ../docs/api ../wepy
# sphinx-build -M html . ../docs

# # do some cleanup
# rm -rf ../docs/doctrees
# mv ../docs/html/* ../docs/
# rm -rf ../docs/html


# also build locally in sphinx for easier testing
make clean
# generate the api rst files for autodoc
sphinx-apidoc -f --separate --module-first -o api ../wepy
sphinx-build -b html -E -a -j 6 . ./_build/html/
