FROM python:3.8

RUN mkdir /wh
WORKDIR /wh

# COPY MLproject .
COPY pyproject.toml .
COPY setup.cfg .
COPY src ./src

RUN ["python3", "-m", "pip", "install", "."]
