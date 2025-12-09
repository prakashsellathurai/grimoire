#!/bin/bash

git add .
commitMessage="Auto commit on `date +%Y%m%d%H%M%S`";
git commit -m "$commitMessage"
git push origin main
