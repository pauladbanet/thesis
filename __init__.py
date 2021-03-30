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

BATCH_SIZE = 32
AUTOTUNE = tf.data.experimental.AUTOTUNE
SLICE_LENGTH = 323

REMOVE_OFFSET_MFCC = True