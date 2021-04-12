#!/bin/bash

MODULE_NAME=trainer.json_main
PACKAGE_NAME=trainer
JOB=test7
BUCKET=gs://job_results

gcloud ai-platform jobs submit training $JOB \
--module-name $MODULE_NAME \
--package-path $PACKAGE_NAME \
--region=us-central1 \
--master-image-uri=gcr.io/paula-309109/thesis \
--job-dir=$BUCKET/$JOB \
--staging-bucket=$BUCKET \
-- \
--user_first_arg 0.01