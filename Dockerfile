# Use a minimal base image with Python 3.10
FROM python:3.10-slim-bullseye

# Instal git, because it's needed for DVC
RUN apt update && apt install -y git

# install DVC
RUN pip install dvc

# Put DVC cache in another directory, so that it doesn't get
# committed to the repository
RUN dvc cache dir --global $HOME/.dvc/cache

# Use this directory as the working directory inside the container
WORKDIR /usr/src/oral-cavity-paper

# Copy the requirements file and install the dependencies
COPY requirements.txt .
RUN pip install -U pip setuptools && pip install -r requirements.txt

# Update the data sources and reproduce the pipeline. For this to work, don't forget to
# provide a volume with the scripts and the pipeline dvc.yaml file. For example:
# docker run \
#     --rm \
#     --volume <host>/oral-cavity-paper:/usr/src/oral-cavity-paper \
#     --name ocp-container \
#     ocp-image
CMD dvc update -R ./data/;dvc repro
