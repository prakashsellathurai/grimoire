#!/bin/bash
commitMessage="Backup Daemon: Auto commit on `date +%Y%m%d%H%M%S`";
git submodule foreach --recursive "git fetch"
git submodule foreach --recursive "git pull origin main"
git submodule foreach --recursive "git add ."
git submodule foreach --recursive \
  "git diff --cached --quiet && git diff --quiet || git commit -a -q -m '${commitMessage}' || :"

git add .
git commit -m "$commitMessage"
git push origin main --recurse-submodules=on-demand
