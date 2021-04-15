# from read_tfrecords import *
# from alden  import *
from trainer import read_tfrecords
from trainer  import alden
from tensorflow.keras import Input
import tensorflow as tf
import os
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D,Conv1D,MaxPooling2D, MaxPooling1D, Activation, Dropout, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

BATCH_SIZE = 32
AUTOTUNE = tf.data.experimental.AUTOTUNE
SLICE_LENGTH = 323
REMOVE_OFFSET_MFCC = False

# export PYTHONPATH=$(pwd)

def get_dataset(file_paths):
  dataset = read_tfrecords.load_dataset(file_paths)
  dataset = dataset.prefetch(buffer_size=AUTOTUNE)
  dataset = dataset.shuffle(buffer_size=500)  
  dataset = dataset.batch(32, drop_remainder=True)
  return dataset


def cnn(opt, input_shape, args):
  input = Input(shape=input_shape)
 
  x = Conv2D(16, (2, 1), activation='relu', kernel_initializer='glorot_normal')(input)
  x = MaxPooling2D(pool_size=(1, 2))(x)
  x = BatchNormalization(renorm=True)(x)

  x = Conv2D(32, (2, 1), activation='relu', kernel_initializer='glorot_normal')(x)
  x = MaxPooling2D(pool_size=(2, 1))(x)
  x = BatchNormalization(renorm=True)(x)

  x = Conv2D(64, (2, 1), activation='relu', kernel_initializer='glorot_normal')(x)
  x = MaxPooling2D(pool_size=(2, 1))(x)
  x = BatchNormalization(renorm=True)(x)
  x = Flatten()(x)
  
  if args.n_dense == 2:
    x = Dense(args.neuron1, activation = 'relu', kernel_initializer='glorot_normal')(x)
    x = Dense(args.neuron2, activation = 'relu', kernel_initializer='glorot_normal')(x)
  if args.n_dense == 3:
    x = Dense(args.neuron1, activation = 'relu', kernel_initializer='glorot_normal')(x)
    x = Dense(args.neuron2, activation = 'relu', kernel_initializer='glorot_normal')(x)
    x = Dense(args.neuron3, activation = 'relu', kernel_initializer='glorot_normal')(x)
  elif args.n_dense == 4:
    x = Dense(args.neuron1, activation = 'relu', kernel_initializer='glorot_normal')(x)
    x = Dense(args.neuron2, activation = 'relu', kernel_initializer='glorot_normal')(x)
    x = Dense(args.neuron3, activation = 'relu', kernel_initializer='glorot_normal')(x)
    x = Dense(args.neuron4, activation = 'relu', kernel_initializer='glorot_normal')(x)
  else:
    x = Dense(512, activation = 'relu', kernel_initializer='glorot_normal')(x)
    x = Dense(512, activation = 'relu', kernel_initializer='glorot_normal')(x)
    x = Dense(256, activation = 'relu', kernel_initializer='glorot_normal')(x)
    x = Dense(64, activation = 'relu', kernel_initializer='glorot_normal')(x)

  x = Dense(1, activation = 'linear')(x)

  x = Model(inputs=input, outputs=x)
  x.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae'])
  
  print(x.summary())
  return x


def lstm(opt, input_shape):
  input = Input(shape=input_shape)
  x = LSTM(10, return_sequences=False)(input)
  x = Dense(16, activation = 'relu')(x)
  x = Dense(1, activation = 'linear')(x)

  x = Model(inputs=input, outputs=x)
  x.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae'])
  
  print(x.summary())
  return x


def start_training(path_train, path_val, args):

  train_dataset = get_dataset(path_train) 
  val_dataset = get_dataset(path_val)            

  for song in train_dataset.take(1):
      input_shape = song[0].shape[1:]

  opt = tf.keras.optimizers.Adam(learning_rate=args.lr)

  model = cnn(opt, input_shape, args)

  checkpoint_path = os.path.join(args.tensorboard_path, 'checkpoint')
  tensorboard_path = os.path.join(args.tensorboard_path, 'logs')

  callback_train = alden.PredictionPlot(tensorboard_path, 'train', train_dataset)
  callback_val = alden.PredictionPlot(tensorboard_path, 'val', val_dataset)
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_path, histogram_freq=1)

  checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='loss', save_best_only=True, mode='auto', period=1)
  # model.load_weights('weights/3conv_5Dense_0.001_adamx2.hdf5')

  callbacks=[tensorboard_callback, callback_train, callback_val, checkpoint]

  hist = model.fit(train_dataset,
                    validation_data=val_dataset,
                    epochs=args.epochs, 
                    callbacks=[checkpoint, tensorboard_callback, callback_train, callback_val])

  return model

# Adam, 0.0001