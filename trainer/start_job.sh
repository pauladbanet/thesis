#!/bin/bash

MODULE_NAME="trainer.json_main"
PACKAGE_NAME="../trainer"
JOB="test18_$(date +%Y%m%d_%H%M%S)"
BUCKET="gs://job_results"
SCALE_TIER='BASIC'
REGION="us-central1"


gcloud ai-platform jobs submit training $JOB \
--config=hyper_config.yaml \
--job-dir $BUCKET/$JOB \
--module-name $MODULE_NAME \
--package-path $PACKAGE_NAME \
--region $REGION \
--staging-bucket $BUCKET \
--runtime-version 2.4 \
--python-version 3.7 \
--scale-tier $SCALE_TIER


# gcloud ai-platform local train  \
# --module-name=$MODULE_NAME \
# --package-path=$PACKAGE_NAME \
# --job-dir $BUCKET/$JOB \
# -- \
# --alpha=0.01
