#!/bin/bash

# syncs given notebook in python format and converts the ipynb file into HTML after running it

# exit on first error
set -e

if [ "$1" == "" ]; then
    echo "Name of the notebook is missing"
    exit
else
    NOTEBOOK=$1
fi

if [ -f $NOTEBOOK ];then
    IPNYB_NOTEBOOK=${NOTEBOOK%%.py}.ipynb

    # if the notebook exists, just sync it (use the newest version)
    # if not, create it
    if [ -f $IPNYB_NOTEBOOK ];then
        jupytext --sync ${NOTEBOOK}
        else jupytext --to notebook ${NOTEBOOK}
    fi

    jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to html --execute ${IPNYB_NOTEBOOK}
else
    echo "${NOTEBOOK} does not exists"
    exit 1
fi
