#!/bin/bash

set -e 

ID=`git rev-parse HEAD`

git checkout gh-pages
rm -rf plotutils build
git rm -rf _*
git checkout master -- doc/_build/html
git mv -f doc/_build/html/* .
git commit -a -m "Docs for commit $ID"
git checkout master
