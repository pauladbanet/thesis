from save_tfrecords import *
from read_tfrecords import *
from keras.models import Sequential
from keras.layers import Dense, Conv2D,Conv1D, MaxPooling2D, Activation, Dropout, Flatten
from keras.optimizers import Adam
from keras.utils import to_categorical

file_path0 = '/dataset/mfccs200_0.tfrecords'
file_path1 = '/dataset/mfccs200_1.tfrecords'
# file_path2 = '/dataset/mfccs200_2.tfrecords'
file_path3 = '/dataset/mfccs200_3.tfrecords'


def get_dataset(file_paths):
  dataset = load_dataset(file_paths)
  dataset = dataset.prefetch(buffer_size=AUTOTUNE)
  dataset = dataset.repeat(100)
  dataset = dataset.shuffle(buffer_size=10 * BATCH_SIZE)  
  dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
  return dataset


def cnn(input_shape):
  model = Sequential()
  model.add(Conv1D(32, 2, activation='relu', input_shape=input_shape))

  model.add(Flatten())

  # model.add(Dense(512, activation='relu'))
  # model.add(Dropout(0.1))

  model.add(Dense(256, activation='relu'))
  model.add(Dense(128, activation='relu'))
  model.add(Dense(3, activation='relu'))

  model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),)
  print(model.summary())
  return model


train_dataset = get_dataset([file_path0, file_path1])
val_dataset = get_dataset([file_path3])

iterator = train_dataset.__iter__()
mfcc_batch, vad_batch = iterator.get_next()

print('mfcc_batch', mfcc_batch.shape, 'vad_batch', vad_batch.shape)
input_shape = mfcc_batch.shape[1:3].as_list()

# Callbacks
initial_learning_rate = 0.01
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate, 
                                                            decay_steps=20, decay_rate=0.96, staircase=True)


model = cnn(input_shape)


hist = model.fit(train_dataset, 
                batch_size=BATCH_SIZE, 
                epochs=100, 
                validation_data=val_dataset)
