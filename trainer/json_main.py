from trainer.model_cnn import *
import argparse
import tensorflow as tf
import pickle5 as pickle
import os
import json
from google.cloud import storage

if data['vgg']:
    tf_files = tf.io.gfile.listdir('gs://vggs')
    tf_paths = ['gs://vggs/' + file for file in tf_files]

    train_path = tf_paths[0:20]
    val_path = tf_paths[20:26]
    test_path = tf_paths[26:32]

    # # For trying small things
    # train_path = tf_paths[0:2]
    # val_path = tf_paths[20]
    # test_path = tf_paths[26]

else:
    tf_files = tf.io.gfile.listdir('gs://mfccs')
    tf_paths = ['gs://mfccs/' + file for file in tf_files]

    train_path = tf_paths[0:20]
    val_path = tf_paths[20:26]
    test_path = tf_paths[26:32]

    # # For trying small things
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

storage_client = storage.Client('paula-309109')
bucket = storage_client.get_bucket('job_results')

if data['test_vgg']:
    # When loading a model already trained
    model = tf.keras.models.load_model('gs://job_results/' + data['name'] +'/model')
    
else:
    # When training and then save the model
    model = start_training(train_path, val_path, args)
    export_module_dir = os.path.join(args.tensorboard_path, 'model')
    tf.keras.models.save_model(model, export_module_dir)

test = get_dataset(test_path, data['batch_size'], test=True)

loss = model.evaluate(test, verbose=0)
print(f'Test loss: {loss[0]} / Test mae: {loss[1]}')

soundtracks6600 = pickle.load(open('/dataset/soundtracks6600.pkl', 'rb'))
dataframes = [soundtracks6600[i:i+200] for i in range(0,soundtracks6600.shape[0],200)]
df_test = dataframes[32]
test_songs = get_dataset('gs://vggs/vgg_32.tfrecords', 1, test=True)

labels_all = []

for item, song in zip(df_test.iterrows(), test_songs):
    print(item[1].id)
    pieces_15 = int(song[0].shape[1] / 15)
    batch_song = []

    # Create batch with number of 15 pieces per song
    for i in range(0, pieces_15):
        piece = tf.slice(song[0], [0, 15*i, 0], [-1, 15, -1])
        batch_song.append(piece)
    
    label_song  = []

    # Predict for each song of the batch
    for j in batch_song:
        test_result = model.predict(j)
        test_result = ' '.join(map(str, test_result)).replace('[', '').replace(']', '')
        label_song.append(test_result)

    labels_all.append(label_song)

blob = bucket.blob(data['name'] + '/val_predicted.json')
blob.upload_from_string(data=json.dumps(labels_all))   
