from save_tfrecords import *
from read_tfrecords import *
from alden import *
from __init__ import *
from keras import Input
from keras.models import Sequential, Model
from keras.layers import Dense, TimeDistributed, Conv2D,Conv1D,MaxPooling2D, MaxPooling1D, Activation, Dropout, Flatten, BatchNormalization, LSTM
from keras.optimizers import Adam
from keras.utils import to_categorical

def get_dataset(file_paths):
  dataset = load_dataset(file_paths)
  dataset = dataset.prefetch(buffer_size=AUTOTUNE)
  dataset = dataset.shuffle(buffer_size=500)  
  dataset = dataset.batch(32, drop_remainder=True)
  return dataset


def lstm(opt, input_shape):
  input = Input(shape=input_shape)
  x = LSTM(10, return_sequences=False)(input)
  x = Dense(16, activation = 'relu')(x)
  x = Dense(1, activation = 'linear')(x)

  x = Model(inputs=input, outputs=x)
  x.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae'])
  
  print(x.summary())
  return x

train_dataset = get_dataset([file_path0, file_path1, file_path2])  # 600 songs
val_dataset = get_dataset([file_path4])             # 200 songs

for song in train_dataset.take(1):
    input_shape = song[0].shape[1:]

lr = 0.001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(lr, decay_steps=20, decay_rate=0.96, staircase=True)
opt = tf.keras.optimizers.Adam(learning_rate=lr)

model = lstm(opt, input_shape)

log_dir = 'logs/lstm/400lstmx1000' + opt._name + str(lr)

callback_train = PredictionPlot(log_dir, 'train', train_dataset)
callback_val = PredictionPlot(log_dir, 'val', val_dataset)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

checkpoint = tf.keras.callbacks.ModelCheckpoint('weights/400lstmx1000' + opt._name + str(lr) +'.hdf5', monitor='loss', save_best_only=True, mode='auto', period=1)
# model.load_weights('weights/x500Adam0.001.hdf5')

hist = model.fit(train_dataset, 
                  validation_data=val_dataset,
                  epochs=1000, 
                  callbacks=[checkpoint, tensorboard_callback, callback_train, callback_val]) # 
