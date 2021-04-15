import sys 
from trainer import model_cnn
# from model_cnn import *
import argparse
import hypertune
import joblib
from google.cloud import storage

num_files = range(0,15)
GC_PATHS = []

for num_file in num_files:
    file_path = 'gs://mfccs/mfccs200_' + str(num_file) + '.tfrecords'
    GC_PATHS.append(file_path)

parser = argparse.ArgumentParser()

parser.add_argument(
    '--lr',
    default=0.001,
    type=float)

parser.add_argument(
    '--epochs',
    default=10,
    type=int)

parser.add_argument(
    '--n_dense',
    default=3,
    type=int)

parser.add_argument(
    '--neuron1',
    default=512,
    type=int)

parser.add_argument(
    '--neuron2',
    default=512,
    type=int)

parser.add_argument(
    '--neuron3',
    default=256,
    type=int)

parser.add_argument(
    '--neuron4',
    default=64,
    type=int)

parser.add_argument(
    '--tensorboard_path',
    default='gs://mfccs/hp_job_dir')

parser.add_argument(
    '--job-dir',
    required=False)
    
args = parser.parse_args()
arguments = args.__dict__
job_dir = arguments.pop('job_dir')

print('lr',args.lr)

model = model_cnn.start_training(GC_PATHS[0:4], GC_PATHS[5], args)

val = model_cnn.get_dataset(GC_PATHS[6])
loss = model.evaluate(val)[0]

# Calling the hypertune library and setting our metric
hpt = hypertune.HyperTune()
hpt.report_hyperparameter_tuning_metric(
    hyperparameter_metric_tag='loss',
    metric_value=loss,
    global_step=args.epochs)














