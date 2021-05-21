import tensorflow as tf
import os
import json
                                                                                                                                                                                                                                                                      
global data
data = {   
    'slice_length': 646, 
    'remove_offset': True,
    'valence': True,
    'arousal': False,
    'dominance': False,
    'lstm': True, 
    'batch_size': 32, 
    'epochs': 1000, 
    'vgg': False,
    'slice_vgg': 15,
    'name': 'mfcc_lstm_val_rem_646', 
    'test_second': 0, 
    'test_vgg': False    # Testing loading a previous trained model or not.
    }

def read_tfrecord(serialized_example):
    features = {
        'id': tf.io.FixedLenFeature([], tf.int64),
        'mfcc': tf.io.FixedLenFeature([], tf.string),
        'val': tf.io.FixedLenFeature([], tf.float32),
        'aro': tf.io.FixedLenFeature([], tf.float32),
        'dom': tf.io.FixedLenFeature([], tf.float32),
        }
    example = tf.io.parse_single_example(serialized_example, features)
    id_song = example['id']
    val = example['val']
    aro = example['aro']
    dom = example['dom']
    mfcc = tf.io.parse_tensor(example['mfcc'], out_type=tf.float32)
    
    new_mfcc = tf.reshape(mfcc, [13, int(len(mfcc)/13)]) 
    print('new_mfcc', new_mfcc)
    # Takes a random slice of mfcc 
    max_offset = int(len(mfcc)/13) - data['slice_length']
    random_offset = tf.random.uniform((), minval=0, maxval=max_offset, dtype=tf.dtypes.int32)

    if data['remove_offset']:
        piece = tf.slice(new_mfcc, [1, random_offset], [-1, data['slice_length']])
    else:
        piece = tf.slice(new_mfcc, [0, random_offset], [-1, data['slice_length']])

    piece = tf.transpose(piece)

    if data['lstm']==False:
        piece = tf.expand_dims(piece, axis=2)  # Remove it for LSTM

    # # Quitar esto y ultimos dos params 
    # piece = new_mfcc
    # len_mfcc = len(mfcc)

    if data['valence']:
        return piece, val
    elif data['arousal']:
        return piece, aro
    elif data['dominance']:
        return piece, dom
    else:
        return piece, val


def read_vgg(serialized_example):
    features = {
        'id': tf.io.FixedLenFeature([], tf.int64),
        'emb': tf.io.FixedLenFeature([], tf.string),
        'val': tf.io.FixedLenFeature([], tf.float32),
        'aro': tf.io.FixedLenFeature([], tf.float32),
        'dom': tf.io.FixedLenFeature([], tf.float32),
        }
    example = tf.io.parse_single_example(serialized_example, features)
    id_song = example['id']
    val = example['val']
    aro = example['aro']
    dom = example['dom']
    emb = tf.io.parse_tensor(example['emb'], out_type=tf.float32)
    
    new_emb = tf.reshape(emb, [int(len(emb)/128), 128]) 
    
    # Takes a random slice of emb 
    max_offset = int(len(emb)/128) - data['slice_vgg']
    random_offset = tf.random.uniform((), minval=0, maxval=max_offset, dtype=tf.dtypes.int32)

    piece = tf.slice(new_emb, [random_offset, 0], [data['slice_vgg'], -1])

    # piece = tf.transpose(piece)
    if data['lstm'] == False:
        piece = tf.expand_dims(piece, axis=2)  # Remove it for LSTM

    if data['valence']:
        return piece, val
    elif data['arousal']:
        return piece, aro
    elif data['dominance']:
        return piece, dom
    else:
        return piece, val


def read_vgg_test_inorder(serialized_example):
    features = {
        'id': tf.io.FixedLenFeature([], tf.int64),
        'emb': tf.io.FixedLenFeature([], tf.string),
        'val': tf.io.FixedLenFeature([], tf.float32),
        'aro': tf.io.FixedLenFeature([], tf.float32),
        'dom': tf.io.FixedLenFeature([], tf.float32),
        }
    example = tf.io.parse_single_example(serialized_example, features)
    id_song = example['id']
    val = example['val']
    aro = example['aro']
    dom = example['dom']
    emb = tf.io.parse_tensor(example['emb'], out_type=tf.float32)
    
    new_emb = tf.reshape(emb, [int(len(emb)/128), 128]) 
    # piece = tf.slice(new_emb, [data['test_second'], 0], [data['slice_vgg'], -1])
    piece = new_emb
    # piece = tf.transpose(piece)
    if data['lstm'] == False:
        piece = tf.expand_dims(piece, axis=2)  # Remove it for LSTM

    if data['valence']:
        return piece, val
    elif data['arousal']:
        return piece, aro
    elif data['dominance']:
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

def load_dataset_vgg(file_paths):
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False 
    dataset = tf.data.TFRecordDataset(file_paths)
    dataset = dataset.with_options(ignore_order)
    if data['test_vgg']:
        dataset = dataset.map(read_vgg_test_inorder)
    else:
        dataset = dataset.map(read_vgg) 
    return dataset


