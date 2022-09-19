# WalletHub Score

This project is intended to predict a value I'm assuming it's a credit score.


## Installation

### Project dependecies

You should have these tools installed din your machine:

- Docker
- Docker compose
- Make

### Creating a virtual envronment

- Run `make venv` to create a new virtual environment
  - It will be created on a `.venv` folder
- Run `make install` to install all dependencies

### Running on docker

You just need to run `docker compose up --build`, this will create the following:
- MlFlow running on `http://localhost:5000`
- Minio S3 artifact store running on `http://localhost:9001`
  - user: admin
  - pwd: admin_key

> For convenience, both `data` folder and `.env` files were versioned in git. It should not be in a real production project.
> This way you can see all the experiment tracking using MlFlow ui in `http://localhost:5000`.
> You alse have the trained model in the artifact store, no you don't need to retrain to get predictions, just need to follow the instructions bellow.

## Making predictions

You shoud have the all the containers running ans the virtual env activated (runnig `. .venv/bin/activate`), then you can call `predict <path to csv>`.
It will generate a file called `predictions.csv` on root folder.

## Retrain the model

You shoud have the all the containers running ans the virtual env activated (runnig `. .venv/bin/activate`), then you can:
- Call `model-train` to run the train locally, or
- Call `make model-train` to run inside the container.
