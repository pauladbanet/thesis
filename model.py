from save_tfrecords import *
from read_tfrecords import *
from keras.models import Sequential
from keras.layers import Dense, Conv2D,Conv1D, MaxPooling1D, Activation, Dropout, Flatten
from keras.optimizers import Adam
from keras.utils import to_categorical

file_path0 = '/dataset/mfccs200_0.tfrecords'
file_path1 = '/dataset/mfccs200_1.tfrecords'
# file_path2 = '/dataset/mfccs200_2.tfrecords'
file_path3 = '/dataset/mfccs200_3.tfrecords'


def get_dataset(file_paths):
  dataset = load_dataset(file_paths)
  dataset = dataset.prefetch(buffer_size=AUTOTUNE)
  # dataset = dataset.shuffle(buffer_size=10 * BATCH_SIZE)  
  dataset = dataset.batch(4, drop_remainder=True)
  return dataset

num_neurons = 30
def cnn():
  model = Sequential()
  model.add(Conv1D(32, 2, activation='relu', input_shape=[13,323]))
  model.add(MaxPooling1D(2))
  # model.add(Dense(num_neurons, activation = 'relu', input_dim = 4199))
  model.add(Flatten())
  model.add(Dense(1 , activation = 'linear'))
  model.compile(loss='mean_squared_error', optimizer='rmsprop')
  print(model.summary())
  return model

  # model = Sequential()
  # model.add(Conv1D(32, 2, activation='relu', input_shape=[13,323]))
  # model.add(MaxPooling1D(2))
  # model.add(Flatten())
  # model.add(Dense(32, activation='relu'))
  # model.add(Dense(32, activation='relu'))
  # model.add(Dense(3, activation='relu'))

  # model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),)
  # print(model.summary())
  # return model

train_dataset = get_dataset([file_path0, file_path1])
val_dataset = get_dataset([file_path3])

initial_learning_rate = 0.01
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate, 
                                                            decay_steps=20, decay_rate=0.96, staircase=True)
model = cnn()
hist = model.fit(train_dataset, 
                  validation_data=val_dataset,
                  epochs=50)
