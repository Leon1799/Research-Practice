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
  # tune multiplier
  global BLEACHING_WEIGHT_MULTIPLIER
  if len(sys.argv) > 1:
    BLEACHING_WEIGHT_MULTIPLIER = float(sys.argv[1])

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


BLEACHING_WEIGHT_MULTIPLIER = 5.0


# Custom BCE loss function that ignores NaNs and upweights higher-bleaching examples (heuristically scaling weights)
def bce(y_true, y_pred):
  y_true, y_pred = nan_mask(y_true, y_pred) # y_true are scaled bleaching fractions and y_pred sigmoid output, flattens to 1D

  # apply multiplication before bce averages individual losses
  y_true_elementwise = tf.expand_dims(y_true, axis=-1) # preserve 1 bce value per label, so weights can be applied to correct labels
  y_pred_elementwise = tf.expand_dims(y_pred, axis=-1)
  bce_per_element = BinaryCrossentropy(reduction='none')(y_true_elementwise, y_pred_elementwise)

  # higher true bleaching fraction -> higher loss weight, since those examples are rare
  weights = 1.0 + BLEACHING_WEIGHT_MULTIPLIER * y_true
  return tf.reduce_mean(bce_per_element * weights)


# Custom metric for accuracy within 10% threshold, ignoring NaNs
def accuracy_at_10(y_true, y_pred):
  y_true, y_pred = nan_mask(y_true, y_pred)
  # Threshold at 0.1 to get arbitrary "bleached" and "non-bleached" classes
  positive_true = y_true > 0.1
  positive_predicted = y_pred > 0.1
  correct_predictions = positive_true == positive_predicted
  return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))


# validation metrics
def calculate_metrics(labels, preds, prefix, threshold=0.1):
  valid = ~np.isnan(labels)
  y_true = labels[valid]
  y_pred = preds[valid]

  error = y_pred - y_true
  positive_true = y_true >= threshold
  positive_pred = y_pred >= threshold

  true_positives = np.sum(positive_true & positive_pred)
  false_positives = np.sum(~positive_true & positive_pred)
  false_negatives = np.sum(positive_true & ~positive_pred)

  # guard against division by zero when a split has no (predicted) positives
  precision = true_positives / max(true_positives + false_positives, 1)
  recall = true_positives / max(true_positives + false_negatives, 1)

  return {
    f'{prefix}_valid_label_count': int(len(y_true)), # pixel level count of validation labels
    f'{prefix}_bleaching_positive_count': int(np.sum(positive_true)), #pixel level count of >10% bleaching labels
    f'{prefix}_masked_mae_percent': float(100 * np.mean(np.abs(error))),
    f'{prefix}_masked_bias_percent': float(100 * np.mean(error)),
    f'{prefix}_bleaching_precision_at_10': float(precision),
    f'{prefix}_bleaching_recall_at_10': float(recall),
  }


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

    # pixel-level metrics
    if logs is not None:
      logs.update(calculate_metrics(labels, preds, prefix='val')) # pyright: ignore[reportAttributeAccessIssue]

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

    # site-level metrics: pixel predictions averaged per site/quarter first, so large sites don't dominate
    if logs is not None:
      site_metrics = calculate_metrics(labels, means, prefix='val_site')
      logs['val_site_masked_mae_percent'] = site_metrics['val_site_masked_mae_percent'] # pyright: ignore[reportIndexIssue]

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
