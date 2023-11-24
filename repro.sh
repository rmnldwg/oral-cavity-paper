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

info "Create and activate virtual environment:"
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
TMP_DIR="$( mktemp --directory )"
eval "$LATEST_PY -m venv $TMP_DIR"
source $TMP_DIR/bin/activate
echo "done"

info "Upgrade pip and setuptools:"
python -m pip install --upgrade pip setuptools

info "Install requirements:"
python -m pip install -r $SCRIPT_DIR/requirements.txt

info "Download/update data sources:"
python -m dvc update -R $SCRIPT_DIR/data/

info "Reproduce pipeline"
python -m dvc repro -f $SCRIPT_DIR/dvc.yaml

deactivate
