#!/bin/bash

MODULE_NAME="trainer.json_main"
PACKAGE_NAME="../trainer"
JOB="lstm_vgg_15_valence"
BUCKET="gs://job_results"
REGION="us-central1"

# gcloud ai-platform jobs submit training $JOB \
# --config=hyper_config.yaml \
# --job-dir $BUCKET/$JOB \
# --module-name $MODULE_NAME \
# --package-path $PACKAGE_NAME \
# --region $REGION \
# --staging-bucket $BUCKET \
# --runtime-version 2.4 \
# --python-version 3.7 \
# -- \
# --tensorboard_path=$BUCKET/$JOB 


gcloud ai-platform local train  \
--module-name=$MODULE_NAME \
--package-path=$PACKAGE_NAME \
--configuration=hyper_config.yaml 
-- \
--tensorboard_path=$BUCKET/$JOB \


# MODEL_NAME="model_test_predict1"

# # Create the model
# gcloud ai-platform models create $MODEL_NAME \
#   --regions $REGION

# JOB_DIR='gs://job_results/cnn_3_0421_1510'
# MODEL_VERSION="${MODEL_NAME}_v1"
# SAVED_MODEL_PATH=$(gsutil ls $JOB_DIR | tail -n 1)

# # Create model version based on that SavedModel directory
# gcloud ai-platform versions create $MODEL_VERSION \
#   --model $MODEL_NAME \
#   --runtime-version 2.4 \
#   --python-version 3.7 \
#   --origin $SAVED_MODEL_PATH


# PREDICTION_NAME='prediction'
# INPUT_PATHS=gs://mfccs/mfccs200_8.tfrecords, gs://mfccs/mfccs200_9.tfrecords
# OUTPUT_PATH="${JOB_DIR}/predictions"

# gcloud ai-platform jobs submit prediction $PREDICTION_NAME \
# --model $MODEL_NAME \
# --input-paths $INPUT_PATHS \
# --output-path $OUTPUT_PATH \
# --region $REGION \
# --data-format 'tf-record' \
# --batch-size 32