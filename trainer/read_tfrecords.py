
import tensorflow as tf
from trainer import model_cnn
import os
import json

def read_tfrecord(serialized_example):
    features = {
        'id': tf.io.FixedLenFeature([], tf.int64),
        'mfcc': tf.io.FixedLenFeature([], tf.string),
        'val': tf.io.FixedLenFeature([], tf.float32),
        'aro': tf.io.FixedLenFeature([], tf.float32),
        'dom': tf.io.FixedLenFeature([], tf.float32),
        }
    example = tf.io.parse_single_example(serialized_example, features)
    id = example['id']
    val = example['val']
    aro = example['aro']
    dom = example['dom']
    mfcc = tf.io.parse_tensor(example['mfcc'], out_type=tf.float32)
    
    new_mfcc = tf.reshape(mfcc, [13, int(len(mfcc)/13)]) 
    
    # Takes a random slice of mfcc 
    max_offset = int(len(mfcc)/13) - model_cnn.data['slice_length']
    random_offset = tf.random.uniform((), minval=0, maxval=max_offset, dtype=tf.dtypes.int32)

    if model_cnn.data['remove_offset']:
        piece = tf.slice(new_mfcc, [1, random_offset], [-1, model_cnn.data['slice_length']])
    else:
        piece = tf.slice(new_mfcc, [0, random_offset], [-1, model_cnn.data['slice_length']])

    piece = tf.transpose(piece)

    if model_cnn.data['lstm']==False:
        piece = tf.expand_dims(piece, axis=2)  # Remove it for LSTM

    # # Quitar esto y ultimos dos params 
    # piece = new_mfcc
    # len_mfcc = len(mfcc)

    if model_cnn.data['valence']:
        return piece, val
    elif model_cnn.data['arousal']:
        return piece, aro
    elif model_cnn.data['dominance']:
        return piece, dom
    else:
        return piece, val


def load_dataset(file_paths):
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False 
    dataset = tf.data.TFRecordDataset(file_paths)

    dataset = dataset.with_options(ignore_order)
    dataset = dataset.map(read_tfrecord)
    return dataset

