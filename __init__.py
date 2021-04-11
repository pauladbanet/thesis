import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pickle5 as pickle
import pandas as pd
import re
import subprocess
import glob
import os 
import csv
import numpy as np
import tensorflow as tf
from mutagen.mp3 import MP3
import datetime
import matplotlib.pyplot as plt
import librosa


BATCH_SIZE = 32
AUTOTUNE = tf.data.experimental.AUTOTUNE
SLICE_LENGTH = 323

REMOVE_OFFSET_MFCC = False

file_path0 = '/dataset/mfccs200_0.tfrecords'
file_path1 = '/dataset/mfccs200_1.tfrecords'
file_path2 = '/dataset/mfccs200_2.tfrecords'
file_path3 = '/dataset/mfccs200_3.tfrecords'
file_path4 = '/dataset/mfccs200_4.tfrecords'
file_path5 = '/dataset/mfccs200_5.tfrecords'
file_path6 = '/dataset/mfccs200_6.tfrecords'
file_path7 = '/dataset/mfccs200_7.tfrecords'

num_files = range(0,15)
GC_PATHS = []

for num_file in num_files:
    file_path = 'gs://mfccs/mfccs200_' + str(num_file) + '.tfrecords'
    GC_PATHS.append(file_path)


LOCAL_PATHS = []

for local_file in num_files:
    file_path = '/dataset/mfccs200_' + str(local_file) + '.tfrecords'
    LOCAL_PATHS.append(file_path)
