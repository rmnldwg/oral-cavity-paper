#!/bin/bash

# Initialize virtual environment, install dependencies, and run the pipeline.
color=$(tput setaf 4)
normal=$(tput sgr0)

info() {
    printf "%s\n" "${color}$1${normal}"
}

info "Fetch latest version of repo:"
git fetch origin
git pull

info "Determined latest Python:"
LATEST_PY=$( find /usr/bin -name python3* -type f -executable | sort -V | tail -1 )
eval "$LATEST_PY --version"

info "Create virtual environment at:"
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
eval "$LATEST_PY -m venv $SCRIPT_DIR/.venv"
BIN=$SCRIPT_DIR/.venv/bin
echo "$SCRIPT_DIR/.venv"

info "Upgrade pip and setuptools:"
eval "$BIN/python -m pip install --upgrade pip setuptools"

info "Install requirements:"
eval "$BIN/python -m pip install -r $SCRIPT_DIR/requirements.txt"

info "Download/update data sources:"
eval "$BIN/python -m dvc update -R $SCRIPT_DIR/data/"

info "Reproduce pipeline"
eval "$BIN/python -m dvc repro $SCRIPT_DIR/dvc.yaml"
