'''Creates the json with the parameters to pass to the job.'''

import sys 
from trainer import model_cnn
import argparse
import hypertune


num_files = range(0,15)
GC_PATHS = []

for num_file in num_files:
    file_path = 'gs://mfccs/mfccs200_' + str(num_file) + '.tfrecords'
    GC_PATHS.append(file_path)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--alpha',
        default=0.001)
    parser.add_argument(
        '--job-dir',
        required=False)
    args = parser.parse_args()

    return args


args = get_args()
# arguments = args.__dict__
# job_dir = arguments.pop('job_dir')

print(args.alpha)
epochs = 3

# for n_dense in [4, 5]:

loss = model_cnn.start_training(GC_PATHS[0:4], GC_PATHS[5], args.alpha, epochs)

# # Calling the hypertune library and setting our metric
# hpt = hypertune.HyperTune()
# hpt.report_hyperparameter_tuning_metric(
#     hyperparameter_metric_tag='loss',
#     metric_value=loss,
#     global_step=epochs)





























