#!/bin/bash
commitMessage="Backup Daemon: Auto commit on `date +%Y%m%d%H%M%S`";
git submodule foreach --recursive "git add ."
git submodule foreach "echo 'Committing changes.'; git commit -a -q -m '${commitMessage}' || :"
git add .
git commit -m "$commitMessage"
git push origin main --recurse-submodules=on-demand
