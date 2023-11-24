# Patterns of Lymph Node Involvement for Oral Cavity Squamous Cell Carcinoma

This repository provides the source code to reproduce all figures and table in our publication with the above title that we submitted to [JAMA Otolaryngology] and update them in case the data changes. Below we provide instructions on how to achieve this.

[JAMA Otolaryngology]: https://jamanetwork.com/journals/jamaotolaryngology


## Reproduce the Pipeline

First, clone the repository:
```
git clone https://github.com/rmnldwg/oral-cavity-paper
```

And change your working directory to the root of the repository:
```
cd oral-cavity-paper
```


### Using `bash`

If you're using `bash` on Linux and don't want to install Docker, you may get away with simply running the `repro.sh` script:
```
bash setup.sh
```

It _should_ do the following things in order:
1. pull changes from the GitHub remote repo,
2. automatically find your latest Python 3 installation,
3. set up a virtual environment,
4. install all Python packages defined in `requirements.txt`,
5. download (and update) the data sources,
6. reproduce the pipeline

But this script is probably quite platform dependent. So, if this does not work, a safer way to get the desired results is using Docker, which is explained below.


### Using Docker

Although Docker is platform independent, it is sometimes a hassle to get it installed and running. Here is a guide to the [Docker installation].

If Docker is up and running on your system, the following two commands should do the reproduction:

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
