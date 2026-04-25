#!/bin/bash
# Switch to your local main branch
git checkout main

# Fetch the latest changes from the original repository
git fetch upstream

# Merge changes into your local main (preserves history)
# OR use 'git rebase upstream/main' for a cleaner history
git rebase upstream/main

# Push the updated local main to your GitHub fork (origin)
git push origin main
