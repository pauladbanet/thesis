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

BATCH_SIZE = 32
AUTOTUNE = tf.data.experimental.AUTOTUNE
SLICE_LENGTH = 323

REMOVE_OFFSET_MFCC = False

file_path0 = '/dataset/mfccs200_0.tfrecords'
file_path1 = '/dataset/mfccs200_1.tfrecords'
file_path2 = '/dataset/mfccs200_2.tfrecords'
file_path3 = '/dataset/mfccs200_3.tfrecords'
file_path4 = '/dataset/mfccs200_4.tfrecords'