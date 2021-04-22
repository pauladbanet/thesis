import sys 
from trainer import model_cnn
# from model_cnn import *
import argparse
import tensorflow as tf
import hypertune
import joblib
from google.cloud import storage
import os

tf_files = tf.io.gfile.listdir('gs://mfccs')
tf_paths = ['gs://mfccs/' + file for file in tf_files]

train_path = tf_paths[0:20]
val_path = tf_paths[20:26]
test_path = tf_paths[26:]

parser = argparse.ArgumentParser()

parser.add_argument(
    '--tensorboard_path', 
    default='gs://job_results')

parser.add_argument(
    '--job-dir',
    required=False)
    
args = parser.parse_args()
arguments = args.__dict__
job_dir = arguments.pop('job_dir')

model = model_cnn.start_training(train_path, val_path, args)

test = model_cnn.get_dataset(test_path, model_cnn.data['batch_size'])

loss = model.evaluate(test, verbose=0)

print(f'Test loss: {loss[0]} / Test mae: {loss[1]}')

export_module_dir = os.path.join(args.tensorboard_path, 'model')

tf.keras.models.save_model(model, export_module_dir)

# # Calling the hypertune library and setting our metric
# hpt = hypertune.HyperTune()
# hpt.report_hyperparameter_tuning_metric(
#     hyperparameter_metric_tag='loss',
#     metric_value=loss,
#     global_step=args.epochs)














