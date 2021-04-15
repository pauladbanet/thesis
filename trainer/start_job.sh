#!/bin/bash

MODULE_NAME="trainer.json_main"
PACKAGE_NAME="../trainer"
JOB="testtensorboard_$(date +%m%d_%H%M)"
BUCKET="gs://job_results"
SCALE_TIER='BASIC'
REGION="us-central1"
SLICE_LENGTH=323
REMOVE_OFFSET_MFCC=False

gcloud ai-platform jobs submit training $JOB \
--config=hyper_config.yaml \
--job-dir $BUCKET/$JOB \
--module-name $MODULE_NAME \
--package-path $PACKAGE_NAME \
--region $REGION \
--staging-bucket $BUCKET \
--runtime-version 2.4 \
--python-version 3.7 \
--scale-tier $SCALE_TIER \
-- \
--tensorboard_path=$BUCKET/$JOB 


# gcloud ai-platform local train  \
# --module-name=$MODULE_NAME \
# --package-path=$PACKAGE_NAME \
# --configuration=hyper_config.yaml 
# --job-dir $BUCKET/$JOB \
# -- \
# --tensorboard_path=$BUCKET/$JOB \

