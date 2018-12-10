rm source/*~* &> /dev/null
for f in $(ls source); do
    echo "converting source/$f to rst"
    pandoc -f org -t rst -o source/${f%.org}.rst source/$f
done;

rm source/*.rst.rst* &> /dev/null

sphinx-apidoc -f -o api ../wepy
sphinx-build -M html . _build
