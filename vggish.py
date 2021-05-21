import tensorflow as tf
import numpy as np
import tensorflow_hub as hub
import librosa
import pickle5 as pickle
import warnings
warnings.filterwarnings("ignore")

sountracks9000 = pickle.load(open('/home/pdbanet/Vionlabs/datasets/sountracks9000_withapport_something.pkl', 'rb'))
sountracks9000 = sountracks9000.dropna()
# sountracks9000['vgg'] = 0

model = hub.load('https://tfhub.dev/google/vggish/1')

def _bytes_feature(value):
    value = tf.io.serialize_tensor(value).numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def create_example(item, embeddings):
    features = {
        'id': tf.train.Feature(int64_list=tf.train.Int64List(value=[item[1].id])),
        'emb': _bytes_feature(tf.reshape(embeddings, [-1])),
        'val': tf.train.Feature(float_list=tf.train.FloatList(value=[item[1].valence_tags])),
        'aro': tf.train.Feature(float_list=tf.train.FloatList(value=[item[1].arousal_tags])),
        'dom': tf.train.Feature(float_list=tf.train.FloatList(value=[item[1].dominance_tags])),
    }
    example_proto = tf.train.Example(features = tf.train.Features(feature=features))
    return example_proto.SerializeToString()

def write_tfrecords(dataframe, filename):
    dataframe['vgg'] = dataframe['vgg'].astype('object')

    with tf.io.TFRecordWriter(filename) as writer:

        for item in dataframe.iterrows():
            audio_path = '/home/pdbanet/Vionlabs/datasets/soundtracks9000/' + str(item[1].id) + '.mp3'
            y, sr = librosa.load(audio_path, sr=16000)
            embeddings = model(y)
            # [None, 128]
            dataframe.at[item[1].name, 'vgg'] = embeddings
            print('song ID ' + str(item[1].name) + str(embeddings.numpy().shape))

            example = create_example(item, embeddings)

            writer.write(example)

def save_vgg(sountracks):    
    batch_size = 200
    start_index = 29
    dataframes = [sountracks[i:i+batch_size] for i in range(0,sountracks.shape[0],batch_size)]

    for index in range(len(dataframes)):
        
        filename = '/home/pdbanet/Vionlabs/datasets/vggish/vgg_'+ str(index) +'.tfrecords'
        if index >= start_index:
            write_tfrecords(dataframes[index], filename)
            print('Batch index', str(index))

save_vgg(sountracks9000)  
