#!/bin/bash

# Copy files from the `figures` directory to a directory defined in the environment
# variable `FIGURES_OUTPUT_DIR`.

# If `FIGURES_OUTPUT_DIR` environment variable not defined, skip.
if [ -z "${FIGURES_OUTPUT_DIR}" ]; then
    echo "Environment variable FIGURES_OUTPUT_DIR is not defined. Skipping."
    exit 0
fi

# The `FIGURES_OUTPUT_DIR` environment variable must be a directory.
if [ ! -d "${FIGURES_OUTPUT_DIR}" ]; then
    echo "Environment variable FIGURES_OUTPUT_DIR must be a directory."
    exit 1
fi

# Copy files from the `figures` directory to the `FIGURES_OUTPUT_DIR` directory.
cp figures/* "$FIGURES_OUTPUT_DIR"
