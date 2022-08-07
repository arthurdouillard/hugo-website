#!/bin/bash

echo -e "\033[0;32mDeploying updates to GitHub...\033[0m"

# Build the project.
# https://github.com/gohugoio/hugo/releases/tag/v0.49.2
/usr/local/bin/hugo-0.49.2/hugo # if using a theme, replace with `hugo -t <YOURTHEME>`

# Go To Public folder
cd public

# Set CNAME
echo "arthurdouillard.com" > CNAME

# Add changes to git.
git add .


# Commit changes.
msg="rebuilding site `date`"
if [ $# -eq 1 ]
  then msg="$1"
fi
git commit -m "$msg"

# Push source and build repos.
git push origin master

# Come Back up to the Project Root
cd ..

git add public/
git commit -m "Update public submodule."
git push origin master
