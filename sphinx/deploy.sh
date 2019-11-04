#!/bin/bash

# we build directly into this repository
# rsync -rav ./_build/html/ salotz@volta.bch.msu.edu:/volume1/web/wepy/
<<<<<<< HEAD
=======

# make sure we have the remote and have fetched the branch
git remote add github git@github.com:ADicksonLab/wepy.git
git checkout --track github/gh-pages

git checkout gh-pages

# copy over the build products
cp ./_build/html/* ../docs/

git checkout master
>>>>>>> master
