##############
# added the dataset to the path so that it can be imported
import sys 
from pathlib import Path
from typing import Any, cast
############

import tensorflow_datasets as tfds
import numpy as np
from data import train_data, val_data  # pyright: ignore[reportAttributeAccessIssue]
from unet1d import UNet1D
import wandb
from wandb.integration.keras import WandbMetricsLogger
wandb.require("core")
import tensorflow as tf
from tensorflow import keras # pyright: ignore[reportMissingModuleSource]
from tensorflow.keras.losses import BinaryCrossentropy # pyright: ignore[reportMissingModuleSource]
from tensorflow.keras.models import Sequential # pyright: ignore[reportMissingModuleSource]
from tensorflow.keras.layers import Conv1D, InputLayer, Dropout, BatchNormalization, Activation # pyright: ignore[reportMissingModuleSource]
from tensorflow.keras.optimizers import Adam, AdamW # pyright: ignore[reportMissingModuleSource]
import matplotlib.pyplot as plt

######################
# added the dataset to the path so that it can be imported 
PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR / "datasets"))
import corals  # pyright: ignore[reportMissingImports]
###################

def main():
  # Setup a very simple model
  # model = Sequential([
  #   #               time,bands
  #   InputLayer(shape=(98, 4)),
  #   Conv1D(256, kernel_size=3, padding='same'),
  #   BatchNormalization(),
  #   Activation('relu'),
  #   Conv1D(256, kernel_size=3, padding='same'),
  #   BatchNormalization(),
  #   Activation('relu'),
  #   Conv1D(256, kernel_size=3, padding='same'),
  #   BatchNormalization(),
  #   Activation('relu'),
  #   Conv1D( 1, kernel_size=1, padding='same', activation='sigmoid'),  # Output layer
  # ])
  model = UNet1D(
      input_shape=(98, 4),
      num_classes=1,
      filters=64
  )

  wandb.init(project='Corals')
  run = cast(Any, wandb.run)#########
  print(f'models/{run.id}.keras')########

  # Initialize a new W&B run
  model.compile(
      optimizer=Adam(),
      loss=bce,
      metrics=[accuracy_at_10])
  model.fit(train_data(),
            epochs=20,
            validation_data=val_data(), callbacks=[VisualizationCallback(), WandbMetricsLogger()])
  model.save(f'models/{run.id}.keras')################### #models dir must exist already


# Filter out nan values from the timeseries
def nan_mask(y_true, y_pred):
  mask = tf.math.logical_not(tf.math.is_nan(y_true))
  y_true = tf.boolean_mask(y_true, mask)
  y_pred = tf.boolean_mask(y_pred, mask)
  return y_true, y_pred


# Custom BCE loss function that ignores NaNs
def bce(y_true, y_pred):
  y_true, y_pred = nan_mask(y_true, y_pred) # y_true are scaled bleaching fractions and y_pred sigmoid output
  bce = BinaryCrossentropy()
  return bce(y_true, y_pred)


# Custom metric for accuracy within 10% threshold, ignoring NaNs
def accuracy_at_10(y_true, y_pred):
  y_true, y_pred = nan_mask(y_true, y_pred)
  # Threshold at 0.1 to get arbitrary "bleached" and "non-bleached" classes
  positive_true = y_true > 0.1
  positive_predicted = y_pred > 0.1
  correct_predictions = positive_true == positive_predicted
  return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))


class VisualizationCallback(keras.callbacks.Callback):
  def __init__(self, ):
    super().__init__()
    self.ds = val_data(yield_dicts=True)

  def on_epoch_end(self, epoch, logs=None):
    preds, sites, labels = [], [], []
    for batch in self.ds:
      sites.append(batch['site'].numpy())
      preds.append(self.model(batch['satellite']).numpy()[..., 0])
      labels.append(batch['label'].numpy()[..., 0])
    sites, labels, preds = map(np.concatenate, (sites, labels, preds))

    fig, ax = plt.subplots(figsize=(5, 5))
    mask = ~np.isnan(labels)
    ax.scatter(labels[mask], preds[mask])
    ax.plot([0, 1], [0, 1], '--', c='gray')
    ax.set_aspect(True)
    ax.set_xlabel('True Bleaching [%]')
    ax.set_ylabel('Predicted Bleaching [%]')
    wandb.log({'scatter': wandb.Image(fig)}, step=epoch)
    plt.close(fig)

    agg_labels = []
    agg_means = []
    agg_p25s = []
    agg_p75s = []
    for site in np.unique(sites):
      mask = sites == site
      label = labels[mask][0]
      nanmask = ~np.isnan(label)
      label = label[nanmask]
      pred = preds[mask][:, nanmask]
      mean = pred.mean(axis=0)
      p25, p75 = np.percentile(pred, [10, 90], axis=0)

      agg_labels.append(label)
      agg_means.append(mean)
      agg_p25s.append(p25)
      agg_p75s.append(p75)

    labels, means, p25s, p75s = \
      map(np.concatenate, (agg_labels, agg_means, agg_p25s, agg_p75s))
    p50s = (p25s + p75s) / 2
    ranges = (p75s - p25s) / 2

    fig, ax = plt.subplots(figsize=(5, 5))
    mask = ~np.isnan(labels)
    ax.scatter(labels, means)
    ax.errorbar(labels, p50s, fmt='none', yerr=ranges)
    ax.plot([0, 1], [0, 1], '--', c='gray')
    ax.set_aspect(True)
    ax.set_xlabel('True Bleaching [%]')
    ax.set_ylabel('Predicted Bleaching [%]')
    wandb.log({'site-means': wandb.Image(fig)}, step=epoch)
    plt.close(fig)



if __name__ == "__main__": 
  main()
