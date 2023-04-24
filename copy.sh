#!/bin/bash

# Copy files from the `figures` and tables from the `tables` directory to a folder
# defined in the environment variable `OUTPUT_DIR`.

# If `OUTPUT_DIR` environment variable not defined, skip.
if [ -z "${OUTPUT_DIR}" ]; then
    echo "Environment variable OUTPUT_DIR is not defined. Skipping."
    exit 0
fi

# The `OUTPUT_DIR` environment variable must be a directory.
if [ ! -d "${OUTPUT_DIR}" ]; then
    echo "Environment variable OUTPUT_DIR must be a directory."
    exit 1
fi

# Create the `figures` and `tables` directories in the `OUTPUT_DIR` directory.
mkdir -p "$OUTPUT_DIR/figures"
mkdir -p "$OUTPUT_DIR/tables"

# Copy files from the `figures` and `tables` directories to the `OUTPUT_DIR` directory.
cp figures/* "$OUTPUT_DIR/figures"
cp tables/* "$OUTPUT_DIR/tables"
