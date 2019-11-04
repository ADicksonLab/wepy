#!/bin/bash

# we build directly into this repository
# rsync -rav ./_build/html/ salotz@volta.bch.msu.edu:/volume1/web/wepy/

# make sure we have the remote and have fetched the branch
git remote add github git@github.com:ADicksonLab/wepy.git
git checkout --track github/gh-pages

git checkout gh-pages

# copy over the build products
cp -r ./_build/html/* ../docs/

# add the files in the docs folder
git add docs/* --force

# commit
git commit -m "Automated commit from deploy.sh"

# push this branch so it gets published
git push github

# go back to the branch you were on
git checkout -

