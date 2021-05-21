from trainer.read_tfrecords import *
from trainer.alden  import *
from tensorflow.keras import Input
import tensorflow as tf
import os
import json
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM,Conv2D,Conv1D,MaxPooling2D, MaxPooling1D, Activation, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

AUTOTUNE = tf.data.experimental.AUTOTUNE
# export PYTHONPATH=$(pwd)

def get_dataset(file_paths, batch_size, test):
  if data['vgg']:
    dataset = load_dataset_vgg(file_paths)
  else:
    dataset = load_dataset(file_paths)  
  if test == False:  
    dataset = dataset.shuffle(buffer_size=500)  
  dataset = dataset.prefetch(buffer_size=AUTOTUNE)
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
  x = Dense(1, activation = 'linear')(x)
  x = Model(inputs=input, outputs=x)
  x.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae'])
  
  print(x.summary())
  return x


def lstm(opt, input_shape):
  input = Input(shape=input_shape)
  x = LSTM(512, return_sequences=False)(input)
  x = Dense(128, activation = 'relu', kernel_initializer='random_normal')(x)
  x = Dense(128, activation = 'relu', kernel_initializer='random_normal')(x)
  x = Dense(64, activation = 'relu', kernel_initializer='random_normal')(x)
  x = Dense(1, activation = 'linear')(x)
  x = Model(inputs=input, outputs=x)
  x.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae'])
  
  print(x.summary())
  return x


def start_training(path_train, path_val, args):

  train_dataset = get_dataset(path_train, data['batch_size'], test=False) 
  val_dataset = get_dataset(path_val, data['batch_size'], test=False)            

  for song in train_dataset.take(1):
      input_shape = song[0].shape[1:]

  opt = tf.keras.optimizers.Adam(learning_rate=0.001)

  if data['lstm']:
    model = lstm(opt, input_shape)
  else:
    model = cnn(opt, input_shape)

  # # Plot network graph
  # img_file = '/dataset/images/lstm.png'
  # tf.keras.utils.plot_model(model, to_file=img_file, show_shapes=True, show_layer_names=True)

  checkpoint_path = os.path.join(args.tensorboard_path, 'checkpoint.ckpt')
  tensorboard_path = os.path.join(args.tensorboard_path, 'logs')

  callback_train = PredictionPlot(tensorboard_path, 'train', train_dataset)
  callback_val = PredictionPlot(tensorboard_path, 'val', val_dataset)
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_path, histogram_freq=1)

  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=300,
                                                    restore_best_weights=True)
  callbacks=[tensorboard_callback, callback_train, callback_val, early_stopping]

  hist = model.fit(train_dataset,
                    validation_data=val_dataset,
                    epochs=data['epochs'], 
                    callbacks=callbacks)
  return model
