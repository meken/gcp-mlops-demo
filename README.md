# gcp-mlops-demo

This is a sample project that illustrates how to use [Vertex AI](https://cloud.google.com/vertex-ai) on GCP for building and running [MLOps workflows](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning#mlops_level_2_cicd_pipeline_automation).

> This sample has been verified with Python 3.7. At the time of this writing there are some issues with the `numpy` dependencies on the most recent versions of Python, which might require some work-arounds.

## Steps to run the code on Vertex AI

There's a number of different ways of running this code base on Vertex AI. This repository contains [Cloud Build](https://cloud.google.com/build) pipelines for automating the package generation and continous training of models, as well as a Vertex AI pipeline (based on [Kubeflow v2](https://www.kubeflow.org/docs/components/pipelines/v2/introduction/)) for training. However, it's also possible to run these steps individually. The following commands will illustrate how to run just the training on Vertex AI.

First you need to create a source distribution for this repository. Assuming that you've cloned this repository and you're at the root of it, run the following command

```shell
python3 setup.py sdist --formats=gztar
```

Once the distribution is created, upload it to GCS

```shell
BUCKET=...  # set to a bucket that's accessible 
gsutil cp dist/*.tar.gz gs://$BUCKET/code/
```

Now configure your target environment and submit the job. We assume that there's training data on GCS.

```shell
LOCATION=...  # which GCP region to use
JOB_NAME=gcp-mlops-demo-job
PKG_NAME=`ls -1t dist | head -n1`  # use the latest generated package
PYTHON_PACKAGE_URIS=$BUCKET/code/$PKG_NAME
ARGUMENTS="--training-data-dir=$BUCKET/data/train,--output-dir=$BUCKET/outputs"
EXECUTOR_IMAGE_URI=europe-docker.pkg.dev/vertex-ai/training/scikit-learn-0-23:latest
gcloud ai custom-jobs create \
  --region=$LOCATION \
  --display-name=$JOB_NAME \
  --python-package-uris=$PYTHON_PACKAGE_URIS \
  --worker-pool-spec=machine-type=n1-standard-4,replica-count=1,executor-image-uri=$EXECUTOR_IMAGE_URI,python-module=trainer.task \
  --args="$ARGUMENTS"
```

## Running things locally

In order to run this code on your local environment, you need to install this repository as an editable package.

Let's start with setting up a virtual environment.

```shell
python3 -m venv .venv
source .venv/bin/activate
```

Now you can install the package which will also install all the dependencies

```shell
pip install -e .[gcp,dev]
```

The extra package `gcp` provides the capability to build and run Kubeflow v2 pipelines on GCP. The `dev` package is for development related tools (`flake8`, `pytest`, `coverage`).

> Note that the training code has no dependencies on GCP or other Google components, so it can be run on any platform. The GCP libraries are configured as _extra_ dependencies and only needed when you want to facilitate running the training code on GCP.

Once everything is installed, you can run the training task on your local environment by calling the right module.

```shell
DATA_DIR=...  # a local folder which contains CSV files
python3 -m trainer.task --training-data-dir=$DATA_DIR
```
