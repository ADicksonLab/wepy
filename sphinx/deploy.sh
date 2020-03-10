#!/bin/bash

# TODO: turn this into an individual target in the tasks
# SNIPPET: deploy via rsync
# rsync -rav ./_build/html/ salotz@volta.bch.msu.edu:/volume1/web/wepy/

# TODO: make this more general
# TODO: move to python code

# delete and create every time
git branch -D gh-pages
git branch gh-pages master

# make sure we have the remote and have fetched the branch
git remote add github git@github.com:salotz/fshank.git || echo "github remote already present"
git push --force github gh-pages
# git checkout --track github/gh-pages

git checkout gh-pages || { echo "aborting deploy"; exit 1; }

# NOTE: we don't use git pull because we are force pushing always
# git pull

# merge the new changes from master
git merge -s recursive -Xtheirs master -m "Automated Merge From Master"

# copy over the build products without adding all the other build
# product junk in the repo

cd ..

# so add the html build
git add sphinx/_build/html/* --force

# then clean out everything including the ignored files
git clean -x -f -e sphinx/_build/html

# then move the html files in git
git mv -f sphinx/_build/html/* ./

# commit
git commit -m "Automated commit from deploy.sh"

# push this branch so it gets published
git push --force github gh-pages

# go back to the branch you were on
git checkout -

