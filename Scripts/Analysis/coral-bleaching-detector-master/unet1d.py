# runs when train.py is run

import tensorflow as tf
from tensorflow.keras import layers, models, Input # pyright: ignore[reportMissingModuleSource]

def UNet1D(input_shape, num_classes, final_activation='sigmoid',
           filters=64, dropout=0.1, kernel_size=3):
  inputs = Input(shape=input_shape)

  # Pad to a length that stays compatible with four downsampling steps.
  padded_inputs = layers.ZeroPadding1D(padding=(0, 10))(inputs)

  init = layers.Conv1D(filters, 3, padding='same')(padded_inputs)

  # Encoder
  c1 = conv_block(init, filters, kernel_size, dropout)
  p1 = layers.MaxPooling1D(pool_size=2)(c1)

  c2 = conv_block(p1, filters * 2, kernel_size, dropout)
  p2 = layers.MaxPooling1D(pool_size=2)(c2)

  c3 = conv_block(p2, filters * 4, kernel_size, dropout)
  p3 = layers.MaxPooling1D(pool_size=2)(c3)

  c4 = conv_block(p3, filters * 8, kernel_size, dropout)
  p4 = layers.MaxPooling1D(pool_size=2)(c4)

  # Bottleneck
  c5 = conv_block(p4, filters * 16, kernel_size, dropout)

  # Decoder
  u6 = layers.UpSampling1D(size=2)(c5)
  u6 = layers.concatenate([u6, c4])
  c6 = conv_block(u6, filters * 8, kernel_size, dropout)

  u7 = layers.UpSampling1D(size=2)(c6)
  u7 = layers.concatenate([u7, c3])
  c7 = conv_block(u7, filters * 4, kernel_size, dropout)

  u8 = layers.UpSampling1D(size=2)(c7)
  u8 = layers.concatenate([u8, c2])
  c8 = conv_block(u8, filters * 2, kernel_size, dropout)

  u9 = layers.UpSampling1D(size=2)(c8)
  u9 = layers.concatenate([u9, c1])
  c9 = conv_block(u9, filters, kernel_size, dropout)

  outputs = layers.Conv1DTranspose(num_classes, 3, padding='same', activation=final_activation)(c9)
  outputs = layers.Cropping1D(cropping=(0, 10))(outputs)

  model = models.Model(inputs, outputs)
  return model

def conv_block(inputs, filters, kernel_size, dropout):
  x = layers.Conv1D(filters, kernel_size, padding='same')(inputs)
  x = layers.BatchNormalization()(x)
  x = layers.ReLU()(x)
  x = layers.Conv1D(filters, kernel_size, padding='same')(x)
  x = layers.BatchNormalization()(x)
  x = layers.ReLU()(x)
  if dropout:
    x = layers.Dropout(dropout)(x)
  return x
