# from read_tfrecords import *
# from alden  import *
from trainer import read_tfrecords
from trainer  import alden
from tensorflow.keras import Input
import tensorflow as tf
import os
import json
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM,Conv2D,Conv1D,MaxPooling2D, MaxPooling1D, Activation, Dropout, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# with tf.io.gfile.GFile('gs://job_results/params.json') as f:
#   data = json.load(f)

data = {   
    "slice_length": 646, 
    "remove_offset": True,
    "valence": False,
    "arousal": False,
    "dominance": True,
    "lstm": False, 
    "batch_size": 32, 
    "epochs": 1000,
    "vgg": False,
    "slice_vgg": 15
}


AUTOTUNE = tf.data.experimental.AUTOTUNE
# export PYTHONPATH=$(pwd)

def get_dataset(file_paths, batch_size):
  dataset = read_tfrecords.load_dataset(file_paths)
  dataset = dataset.prefetch(buffer_size=AUTOTUNE)
  dataset = dataset.shuffle(buffer_size=500)  
  dataset = dataset.batch(batch_size, drop_remainder=True)
  return dataset


def cnn(opt, input_shape):
  input = Input(shape=input_shape)

  x = BatchNormalization(renorm=True)(input)
  x = Conv2D(16, (2, 1), activation='relu', kernel_initializer='random_normal')(x)
  x = MaxPooling2D(pool_size=(1, 2))(x)
  # x = BatchNormalization(renorm=True)(x)

  x = BatchNormalization(renorm=True)(x)
  x = Conv2D(32, (2, 1), activation='relu', kernel_initializer='random_normal')(x)
  x = MaxPooling2D(pool_size=(2, 1))(x)

  # x = BatchNormalization(renorm=True)(x)
  # x = Conv2D(64, (2, 1), activation='relu', kernel_initializer='glorot_normal')(x)
  # x = MaxPooling2D(pool_size=(2, 1))(x)  
  x = Flatten()(x)
  x = Dense(128, activation = 'relu', kernel_initializer='random_normal')(x)
  x = Dense(128, activation = 'relu', kernel_initializer='random_normal')(x)
  x = Dense(64, activation = 'relu', kernel_initializer='random_normal')(x)

  if data['valence'] | data['arousal'] | data['dominance']:
    x = Dense(1, activation = 'linear')(x)
  else:
    x = Dense(3, activation = 'linear')(x)

  x = Model(inputs=input, outputs=x)
  x.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae'])
  
  print(x.summary())
  return x


def lstm(opt, input_shape):
  input = Input(shape=input_shape)
  x = LSTM(500, return_sequences=False)(input)
  x = Dense(256, activation = 'relu')(x)
  x = Dense(128, activation = 'relu')(x)
  x = Dense(64, activation = 'relu')(x)

  if data['valence'] | data['arousal'] | data['dominance']:
    x = Dense(1, activation = 'linear')(x)
  else:
    x = Dense(3, activation = 'linear')(x)

  x = Model(inputs=input, outputs=x)
  x.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae'])
  
  print(x.summary())
  return x


def start_training(path_train, path_val, args):

  train_dataset = get_dataset(path_train, data['batch_size']) 
  val_dataset = get_dataset(path_val, data['batch_size'])            

  for song in train_dataset.take(1):
      input_shape = song[0].shape[1:]
      print(input_shape)

  opt = tf.keras.optimizers.Adam(learning_rate=0.001)

  if data['lstm']:
    model = lstm(opt, input_shape)
  else:
    model = cnn(opt, input_shape)

  checkpoint_path = os.path.join(args.tensorboard_path, 'checkpoint.ckpt')
  tensorboard_path = os.path.join(args.tensorboard_path, 'logs')

  callback_train = alden.PredictionPlot(tensorboard_path, 'train', train_dataset)
  callback_val = alden.PredictionPlot(tensorboard_path, 'val', val_dataset)
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_path, histogram_freq=1)

  # checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
  #                                                 monitor='val_loss',
  #                                                 save_best_only=True,
  #                                                 mode='min')

  early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                                    patience=300,
                                                    restore_best_weights=True)

  if data['valence'] | data['arousal'] | data['dominance']:
    callbacks=[tensorboard_callback, callback_train, callback_val, early_stopping]
  else:
    callbacks=[tensorboard_callback, early_stopping]

  hist = model.fit(train_dataset,
                    validation_data=val_dataset,
                    epochs=data['epochs'], 
                    callbacks=callbacks)

  return model
