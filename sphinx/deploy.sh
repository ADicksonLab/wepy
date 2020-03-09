#!/bin/bash

# TODO: turn this into an individual target in the tasks
# SNIPPET: deploy via rsync
# rsync -rav ./_build/html/ salotz@volta.bch.msu.edu:/volume1/web/wepy/

# TODO: make this more general
# TODO: move to python code

# make sure we have the remote and have fetched the branch
git remote add github git@github.com:ADicksonLab/wepy.git || echo "github remote already present"
git checkout --track github/gh-pages

git checkout gh-pages || { echo "aborting deploy"; exit 1; }

# NOTE: we don't use git pull because we are force pushing always
# git pull

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
git push --force github gh-pages

# go back to the branch you were on
git checkout -

