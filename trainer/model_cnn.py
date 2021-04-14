from trainer import read_tfrecords
from trainer  import alden

from keras import Input
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D,Conv1D,MaxPooling2D, MaxPooling1D, Activation, Dropout, Flatten, BatchNormalization
from keras.optimizers import Adam
from keras.utils import to_categorical

BATCH_SIZE = 32
AUTOTUNE = tf.data.experimental.AUTOTUNE

# export PYTHONPATH=$(pwd)

def get_dataset(file_paths):
  dataset = read_tfrecords.load_dataset(file_paths)
  dataset = dataset.prefetch(buffer_size=AUTOTUNE)
  dataset = dataset.shuffle(buffer_size=500)  
  dataset = dataset.batch(32, drop_remainder=True)
  return dataset


def cnn(opt, input_shape):
  input = Input(shape=input_shape)
  # x = BatchNormalization(renorm=True)(input)
  
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
  x = Dense(512, activation = 'relu', kernel_initializer='glorot_normal')(x)
  x = Dense(512, activation = 'relu', kernel_initializer='glorot_normal')(x)
  x = Dense(256, activation = 'relu', kernel_initializer='glorot_normal')(x)
  # x = Dense(128, activation = 'relu', kernel_initializer='glorot_normal')(x)
  x = Dense(64, activation = 'relu', kernel_initializer='glorot_normal')(x)
  x = Dense(1, activation = 'linear')(x)

  x = Model(inputs=input, outputs=x)
  x.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae'])
  
  print(x.summary())
  return x


def start_training(path_train, path_val, lr=0.001, epochs=10):

  train_dataset = get_dataset(path_train) 
  val_dataset = get_dataset(path_val)            

  for song in train_dataset.take(1):
      input_shape = song[0].shape[1:]

  opt = tf.keras.optimizers.Adam(learning_rate=lr)

  model = cnn(opt, input_shape)

  # log_dir = 'logs/3Conv5Dense_0.001x2000val' + opt._name

  # callback_train = alden.PredictionPlot(log_dir, 'train', train_dataset)
  # callback_val = alden.PredictionPlot(log_dir, 'val', val_dataset)
  # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

  # checkpoint = tf.keras.callbacks.ModelCheckpoint('weights/3Conv5Dense_0.001x2000val' + opt._name +'.hdf5', monitor='loss', save_best_only=True, mode='auto', period=1)
  # model.load_weights('weights/3conv_5Dense_0.001_adamx2.hdf5')
  # callbacks=[tensorboard_callback, callback_train, callback_val, checkpoint]

  hist = model.fit(train_dataset, 
                    validation_data=val_dataset,
                    epochs=epochs)

  return hist.history['loss']

# Adam, 0.0001