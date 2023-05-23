#!/bin/bash

# Initialize virtual environment, install dependencies pre-commit hooks.
color=$(tput setaf 4)
normal=$(tput sgr0)

info() {
    printf "%s\n" "${color}$1${normal}"
}

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

info "Create virtual environment inside $SCRIPT_DIR/.venv"
python3.10 -m venv $SCRIPT_DIR/.venv
BIN=$SCRIPT_DIR/.venv/bin

info "Upgrade pip and setuptools:"
eval "$BIN/python -m pip install --upgrade pip setuptools"

info "Install requirements:"
eval "$BIN/python -m pip install -r $SCRIPT_DIR/requirements.txt"

info "Install pre-commit hooks:"
eval "$BIN/pre-commit install"
