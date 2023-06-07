# Oral Cavity Paper

_Paper on lymphatic involvement pattern data of oral cavity squamous cell carcinomas._

## Content

- [Quick Setup](#quick-setup)
- [Content](#content)
- [Repository Structure](#repository-structure)
- [Setup](#setup)
- [Running the Pipeline](#running-the-pipeline)
- [Adding New Stages to the Pipeline](#adding-new-stages-to-the-pipeline)


## Quick Setup

To quickly get and install everything necessary to reproduce the pipeline (or add scripts and plots to the pipeline), simply run the `setup.sh` script at the root of this repository:

```
bash setup.sh
```

## Reproducing Figures & Tables using Docker

> :warning: **Note** \
> You need to have a working [docker installation] for this.

After having cloned the repository (`git clone https://github.com/rmnldwg/oral-cavity-paper`) and changed you working directory into it (`cd oral-cavity-paper`), build the docker image:

```
docker build -t ocr-image .
```

and then run it with the following command:

```
docker run \
    --rm --volume .:/usr/src/oral-cavity-paper \
    --name ocp-container \
    ocp-image
```

That should run the pipeline from joining the datasets to plotting the figures and tables.

[docker installation]: https://docs.docker.com/get-docker/


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


## Setup

To make your local setup ready to reproduce everything or start adding yourown scripts, a couple of things are necessary and some are optional, but nice.

When using [`venv`], this is as straightforward as running the `setup.sh` script:

```
bash ./setup.sh
```

This does the following things in order:

1. Create a folder named `.venv` inside the repository where it places the executables for Python 3.10 and the correpsonding `pip`.
2. Install all the required Python packages defined in `requirements.txt`.
3. Run `pre-commit install` to enable so-called [git commit hooks] that are defined in the file `.pre-commit-config.yaml`. They are trigged with every git commit and do some housekeeping stuff, like making sure there is no trailing whitespace anywhere to be found.

> :warning: **Note:** \
> When the `setup.sh` script fails on your machine and/or you use a different virtual environment provider, then - insstead of running the script - follow the steps below

1. Activate your virtual environment of choice. As an example, I will be using [conda] now:

   ```
   $ conda activate myenv
   ```

2. Get the path to your Python executable:

   ```
   $ which python
   /path/to/your/python
   ```

3. Create a symbolic link from `.venv/bin/python` to the path of your Python executable:

   ```
   $ ln -s /path/to/your/python .venv/bin/python
   ```

Now you should see a file at the location `.venv/bin/python` and if you execute it, it should behave like the executable inside your virtual environment. This means that, _inside_ your virtual environment, the command

```
python --version
```

should produce the same output as

```
.venv/bin/python --version
```

_outside_ your environment. This is because your computer actually executes the same Python file.


[`venv`]: https://docs.python.org/3.10/library/venv.html
[conda]: https://docs.conda.io/en/latest/miniconda.html


## Running the Pipeline

Before running the pipeline, you can define the environemnt variable `OUTPUT_DIR` to tell the final stage in the `dvc.yaml` file where to place the created plots. For example:

```bash
export OUTPUT_DIR="/path/to/shared/figures-or-tables/folder"
```

This can be used to automatically put the latest figures in our shared Teams folder. If you do not set the variable, DVC will skip this step with a warning.

Now, to reproduce the pipeline, simply run

```
dvc repro
```

which runs the pipeline and hopefully produces all the plots (and perhaps copies them over to the defined output location).


## Adding New Stages to the Pipeline

The pipeline is a _Directed Acyclic Graph_ that is defined inside the `dvc.yaml` file. Every entry under `stages` in that file defines a node in that graph.

That stage can have any name, but underneath it (2 spaces of indentation) it must contain the following three keys:

1. `cmd`: Here you put the exact command that you would enter in your shell to e.g. run a script. This command will be exeuted to reproduce the step.
2. `deps`: This is a list of dependencies. It specifies which files are necessary to run the defined `cmd`. Hence, this consists of any input files as well as parameter files and the executed script itself. For example:

   ```yaml
   deps:
     - scripts/compute.py
     - data/input.csv
   ```
3. `outs`: Another list, but of the created/modified files. Essentially, it lists what files are the result of executing the `cmd`:

   ```yaml
   outs:
     - figures/nice_plot.png
     - tables/super_numbers.csv
   ```

For a more comprehensive and detailed explanation about this, look at the [docs for `dvc.yaml` files].

[docs for `dvc.yaml` files]: https://dvc.org/doc/user-guide/project-structure/dvcyaml-files#dvcyaml
