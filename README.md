# Oral Cavity Paper

_Paper on lymphatic involvement pattern data of oral cavity squamous cell carcinomas._

## Quick Setup

To quickly get and install everything necessary to reproduce the pipeline (or add scripts and plots to the pipeline), simply run the `setup.sh` script at the root of this repository:

```
bash setup.sh
```

## Repository Structure

- The `requirements.txt` file lists all Python packages necessary to recreate the plots and figures. If you write a new script that uses a library not listed here, add it to the requirements file.
- Inside `scripts` you should put any (Python) scripts that generate figures and plots. It should load data from the `data` directory and place its outputs inside the `figures` folder. It also contains the `.mplstyle` file which can be used to enforce a consistent matplotlib styling for the figures.
- The `data` directory contains the data that is visualized in our plots and figures. When you need to add data to this folder, there are two ways:

    1. If the data is published in a public repository, then you can run the command
        ```
        dvc import <https://url.to/repository> <path/to/file/within/repository> \
            --out data/name-of-data-file.csv \
            --file data/name-of-data-file.csv.dvc
        ```
        to pull the file from that repository and create a permanent reference to it in the form of a `*.dvc` file inside the `data` directory.
    2. When you manually add files to the `data` folder and they are not large binary files, then you can also just track them using git.

- In the `figures` folder, we store the produced plots and figures. Ideally, nothing is put there by hand.
- A pipeline is defined in the `dvc.yaml` file. Every `stage` in this file defines a job, including its dependencies (like input data) and what it produces (e.g. a figure). When you add a script to the `scripts` folder, you should also add a stage to this YAML file, defining its inputs and output files and what command runs the script. You can find more information on how this works in the [DVC documentation].

[DVC documentation]: https://dvc.org/doc

Additionally, the `params.yaml` defines some necessary parameters, e.g. for the preprocessing of the data. This can be used for configuration settings that get loaded by a script, but these can also simply be defined as `GLOBAL_VARIABLES` at the beginning of the script.

## Running the Pipeline

Before running the pipeline, you can define the environemnt variable `FIGURES_OUTPUT_DIR` to tell the final stage in the `dvc.yaml` file where to place the created plots. For example:

```bash
export FIGURES_OUTPUT_DIR="/path/to/shared/figures/folder"
```

This can be used to automatically put the latest figures in our shared Teams folder. If you do not set the variable, [DVC] will skip this step with a warning.

Now, to reproduce the pipeline, first run

```
dvc update --recursive data
```

thereby downloading all externally stored data. Afterwards, executing

```
dvc repro
```

runs the pipeline and hopefully produces all the plots (and perhaps copies them over to the defined output location).
