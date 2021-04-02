from save_tfrecords import *
from read_tfrecords import *
from alden import *
from __init__ import *
from keras.models import Sequential
from keras.layers import Dense, Conv2D,Conv1D,MaxPooling2D, MaxPooling1D, Activation, Dropout, Flatten, BatchNormalization
from keras.optimizers import Adam
from keras.utils import to_categorical

def get_dataset(file_paths):
  dataset = load_dataset(file_paths)
  dataset = dataset.prefetch(buffer_size=AUTOTUNE)
  dataset = dataset.shuffle(buffer_size=500)  
  dataset = dataset.batch(24, drop_remainder=True)
  return dataset

num_neurons = 30
def cnn(opt):
  model = Sequential()
  model.add(BatchNormalization(renorm=True, input_shape=[323, 13, 1]))
  model.add(Conv2D(32, (2, 13), activation='relu', kernel_initializer='glorot_normal'))
  model.add(MaxPooling2D(pool_size=(2, 1)))

  model.add(Flatten())
  model.add(Dense(num_neurons, activation = 'relu', kernel_initializer='glorot_normal'))
  model.add(Dense(1 , activation = 'linear'))

  model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae'])
  
  print(model.summary())
  return model

train_dataset = get_dataset([file_path0, file_path1, file_path2, file_path3])  # 800 songs
val_dataset = get_dataset([file_path4])   # 200 songs


lr = 0.0001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(lr, decay_steps=20, decay_rate=0.96, staircase=True)
opt = tf.keras.optimizers.RMSprop(learning_rate=lr)

model = cnn(opt)

log_dir = "logs/fit/" + str(num_neurons) + 'neuron_' + opt._name + '_batchwithloadweights'

callback_train = PredictionPlot(log_dir, 'train', train_dataset)
callback_val = PredictionPlot(log_dir, 'val', val_dataset)

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

checkpoint = tf.keras.callbacks.ModelCheckpoint('best_weights.hdf5', monitor='loss', save_best_only=True, mode='auto', period=1)

model.load_weights('logs/fit/30neuron_Adam_batch.hdf5')
hist = model.fit(train_dataset, 
                  validation_data=val_dataset,
                  epochs=1000, 
                  callbacks=[tensorboard_callback, callback_train, callback_val, checkpoint])
