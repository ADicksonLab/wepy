#!/bin/bash

# we build directly into this repository
# rsync -rav ./_build/html/ salotz@volta.bch.msu.edu:/volume1/web/wepy/

# make sure we have the remote and have fetched the branch
git remote add github git@github.com:ADicksonLab/wepy.git
git checkout --track github/gh-pages

git checkout gh-pages

# merge the new changes from master
git merge master

# then remove the modules so we can actually build the docs without gh
# pages complaining
rm ../.gitmodules
rm -rf ../wepy-tests

git add ../.gitmodules
git add ../wepy-tests

# copy over the build products
cp -rf ./_build/html/* ../
rm -rf ./_build/html/*

# add the files in the docs folder
git add ../* --force

# commit
git commit -m "Automated commit from deploy.sh"

# push this branch so it gets published
git push github/gh-pages

# go back to the branch you were on
git checkout -

