#!/bin/bash

MODULE_NAME="trainer.json_main"
PACKAGE_NAME="../trainer"
JOB=test13
BUCKET=gs://job_results

# gcloud ai-platform jobs submit training $JOB \
# --config=config.json \
# --module-name=$MODULE_NAME \
# --package-path=$PACKAGE_NAME \
# --region=us-central1 \
# --job-dir=$BUCKET/$JOB \
# --runtime-version 2.4 \
# --python-version 3.7 \
# -- \
# --learning=0.01

gcloud ai-platform local train  \
--module-name=$MODULE_NAME \
--package-path=$PACKAGE_NAME \
-- \
--learning=0.01