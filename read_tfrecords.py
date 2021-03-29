from google.cloud import storage
from __init__ import *
from save_tfrecords import *

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

    return mfcc, [val, aro, dom]

file_path = '/dataset/mfccs200_0.tfrecords'
file_paths = [file_path] 


def load_dataset(file_paths):
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False 
    dataset = tf.data.TFRecordDataset(file_paths)

    dataset = dataset.with_options(ignore_order)
    dataset = dataset.map(read_tfrecord)

    return dataset

# dataset = load_dataset(file_paths)
# for data in dataset.take(2):
#     mfcc = data[0].numpy().reshape(13, int(len(data[0])/13))
#     vad = data[1].numpy()
#     print(data)