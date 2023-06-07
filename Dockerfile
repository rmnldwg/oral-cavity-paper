FROM python:3.10-slim-bullseye
RUN apt update && apt install -y git

WORKDIR /usr/src/oral-cavity-paper

COPY requirements.txt .
RUN pip install -U pip setuptools && pip install -r requirements.txt

CMD ["dvc", "repro", "--downstream", "join"]
