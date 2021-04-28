import sys 
from trainer import model_cnn
# from model_cnn import *
import argparse
import tensorflow as tf
import hypertune
import joblib
from google.cloud import storage
import os
import json

# if model_cnn.data['vgg']:
#     tf_files = tf.io.gfile.listdir('gs://vggs')
#     tf_paths = ['gs://vggs/' + file for file in tf_files]

#     train_path = tf_paths[0:8]
#     val_path = tf_paths[8:10]
#     test_path = tf_paths[10:12]
# else:
tf_files = tf.io.gfile.listdir('gs://mfccs')
tf_paths = ['gs://mfccs/' + file for file in tf_files]

train_path = tf_paths[0:20]
val_path = tf_paths[20:26]
test_path = tf_paths[26:32]

# train_path = tf_paths[0:2]
# val_path = tf_paths[20]
# test_path = tf_paths[26]

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

test = model_cnn.get_dataset(test_path, model_cnn.read_tfrecords.data['batch_size'])

loss = model.evaluate(test, verbose=0)
print(loss)
print(f'Test loss: {loss[0]} / Test mae: {loss[1]}')

export_module_dir = os.path.join(args.tensorboard_path, 'model')

tf.keras.models.save_model(model, export_module_dir)

for song in test:       
    test_result = model.predict(song[0])            
    print('test_result', test_result)   # len 32
    print('real_value song[1]', song[1])
    # print('real_value song[0]', song[0])    # piece mfcc song


# # Calling the hypertune library and setting our metric
# hpt = hypertune.HyperTune()
# hpt.report_hyperparameter_tuning_metric(
#     hyperparameter_metric_tag='loss',
#     metric_value=loss,
#     global_step=args.epochs)














