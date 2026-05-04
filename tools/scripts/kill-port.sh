#!/bin/bash
#

if [ -z "$1" ]; then
    echo -e "Must pass the portnumber\nUsage: kill-port.sh [port-number]"
    exit 1
else
    PID=$(lsof -t -i:"$1")
    if [ -z "$PID" ]; then
        echo "No process found on port $1"
        exit 1
    fi
    kill -9 "$PID" && echo "Killed process $PID on port $1"
fi