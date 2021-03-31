from __init__ import *
import librosa
import matplotlib.pyplot as plt
from timeit import default_timer as timer

import warnings
warnings.filterwarnings("ignore")

REMOVE_FIRST_COEF = False  

sountracks1000 = pickle.load(open('/dataset/sountracks1000.pkl', 'rb'))
sountracks10 = sountracks1000.head(10)

sountracks10['mfcc'] = 0
sountracks1000['mfcc'] = 0

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _bytes_feature(value):
    value = tf.io.serialize_tensor(value).numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def create_example(item, mfcc):
    features = {
        'id': _int64_feature(item[1].id),
        'mfcc': _bytes_feature(mfcc.reshape(-1)),
        'val': _floats_feature(item[1].valence_tags),
        'aro': _floats_feature(item[1].arousal_tags),
        'dom': _floats_feature(item[1].dominance_tags),
    }
    example_proto = tf.train.Example(features = tf.train.Features(feature=features))
    return example_proto.SerializeToString()

def write_tfrecords(dataframe, filename):

    with tf.io.TFRecordWriter(filename) as writer:

        for item in dataframe.iterrows():
            print('song ID ' + str(item[1].name))

            audio_path = '/dataset/soundtracks1000/' + str(item[1].id) + '.mp3'
            x , sr = librosa.load(audio_path)
            mfcc = librosa.feature.mfcc(x, sr=sr, n_mfcc=13, hop_length=1024)

            dataframe['mfcc'] = dataframe['mfcc'].astype('object')
            dataframe.at[item[1].name, 'mfcc'] = mfcc

            # Remove first coefficient MFCC
            if REMOVE_FIRST_COEF:
                index = np.where(mfcc[0])
                mfcc = np.delete(mfcc, index)

            example = create_example(item, mfcc)

            writer.write(example)


def save_songs(sountracks1000):    
# Creates tfrecords files with 200 songs in each one

    batch_size = 200
    start_index = 4
    dataframes = [sountracks1000[i:i+batch_size] for i in range(0,sountracks1000.shape[0],batch_size)]

    for index in range(len(dataframes)):
        
        filename = '/dataset/mfccs200_'+ str(index) +'.tfrecords'
        if index == start_index:
            write_tfrecords(dataframes[index], filename)
            print('Batch index', str(index))


# save_songs(sountracks1000)        


# sountracks1000.to_pickle(os.path.join(dir, 'mfccs/sountracks1000.pkl'))

# for index in range(len(list_df)):
#   print('index', str(index))
#   if index >= start_index:

# file_name = "sountracks1000_frames_30s.pkl"
# open_file = open(file_name, "wb")
# pickle.dump(df_final, open_file)
# open_file.close()
# print('ja')
