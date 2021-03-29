from save_tfrecords import *
from read_tfrecords import *
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D

BATCH_SIZE = 32
AUTOTUNE = tf.data.experimental.AUTOTUNE

def get_dataset(file_paths):
  dataset = load_dataset(file_paths)
  dataset = dataset.prefetch(buffer_size=AUTOTUNE)
  dataset = dataset.batch(BATCH_SIZE)

  return dataset


def get_cnn(input_dim):
  model = Sequential()
  model.add(Dense(20, input_dim=10, kernel_initializer='he_uniform', activation='relu'))
  model.add(Dense(3))

  optimizer = tf.keras.optimizers.RMSprop(lr=0.01)
  model.compile(loss='mae', optimizer=optimizer, metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
  return model

file_path = '/dataset/mfccs200_0.tfrecords'
file_paths = [file_path] 
dataset = get_dataset(file_paths)
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

mfcc_batch, vad_batch = next(iter(dataset))

extract_song(dataset, 2)

model = get_cnn()
print(model.summary())

model.fit(dataset, batch_size=BATCH_SIZE, steps_per_epoch=200//BATCH_SIZE, epochs=2)
