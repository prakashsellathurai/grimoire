#!/bin/bash

FOO_DIR=$PWD

_trigger_compgen_filenames() {
    local cur="$1"

    # Files, excluding directories:
    grep -v -F -f <(compgen -d -P ^ -S '$' -- $FOO_DIR"$cur") \
        <(compgen -f -P ^ -S '$' -- $FOO_DIR"$cur") |
        sed -e 's|^\^'$FOO_DIR'||' -e 's/\$$/ /'

    # Directories:
    compgen -d -S / -- $FOO_DIR"$cur" | sed -e 's|'$FOO_DIR'||'
}
trigger() {
    echo "Running: $@"
    # or actually exec the script, e.g.:
    bash "$FOO_DIR/$1" "${@:2}"
}
_trigger_complete() {
    local cur=${COMP_WORDS[COMP_CWORD]}
    COMPREPLY=( $(_trigger_compgen_filenames "$cur") )
}

complete -o nospace -F _trigger_complete trigger
