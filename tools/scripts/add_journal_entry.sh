#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(pwd)"
JOURNAL_FILE="$SCRIPT_DIR/journal.txt"

if [[ $# -gt 0 ]]; then
  ENTRY="$*"
else
  read -r -p "Journal entry: " ENTRY
fi

if [[ -z "${ENTRY// }" ]]; then
  echo "Entry is empty. Nothing written."
  exit 1
fi

TIMESTAMP="$(date '+%Y-%m-%d')"

{
  echo -ne "\n[$TIMESTAMP] $ENTRY"
} >> "$JOURNAL_FILE"

echo "Saved entry to $JOURNAL_FILE"