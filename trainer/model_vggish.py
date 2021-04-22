import tensorflow as tf
import numpy as np
import tensorflow_hub as hub
import librosa
# sountracks9000 = pickle.load(open('/home/pdbanet/Vionlabs/datasets/sountracks9000.pkl', 'rb'))

model = hub.load('https://tfhub.dev/google/vggish/1')

audio_path = '/dataset/soundtracks9000/9002.mp3'

embeddings = model(audio_path)
embeddings.shape.assert_is_compatible_with([None, 128])
