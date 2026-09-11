# added the dataset to the path so that it can be imported
import sys 
from pathlib import Path
from typing import Any, cast
import tensorflow_datasets as tfds
import numpy as np
from data import train_data, val_data, class_weights  # pyright: ignore[reportAttributeAccessIssue]
try:
  from data import test_data  # pyright: ignore[reportAttributeAccessIssue]
except ImportError:
  test_data = None # if running on 12 site test files
from unet1d import UNet1D
import wandb
from wandb.integration.keras import WandbMetricsLogger
wandb.require("core")
import tensorflow as tf
from tensorflow import keras # pyright: ignore[reportMissingModuleSource]
from tensorflow.keras.optimizers import Adam # pyright: ignore[reportMissingModuleSource]
import coral_ordinal as coral # included specific github commit (see requirements) to have corn included and required tensorflow/python version for this project

import matplotlib.pyplot as plt

# added the dataset to the path so that it can be imported 
PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR / "datasets"))
import corals  # pyright: ignore[reportMissingImports]

USE_CLASS_WEIGHTS = False #for unweighted comparison

NUM_CLASSES = 4
CLASS_WEIGHTS = (
    tf.constant(class_weights(), dtype=tf.float32)
    if USE_CLASS_WEIGHTS
    else tf.ones(NUM_CLASSES, dtype=tf.float32)
) # Store for efficient indexing of class weights during loss computation


# CORN logits to class labels 
def decode_corn_logits(logits):
    cumulative_probs = coral.corn_cumprobs(logits) # cumulative probabilities for each class (product of conditional probabilities: probabilities between 0 and 1 derived from logits)
    return coral.cumprobs_to_label(
        cumulative_probs,
        threshold=0.5,
    ) # if > 0.5 assign to class


# Flatten labels and logits and ignore NAN in loss
# satellite: [B,98,4bands], labels: [B,98,1], logits: [B,98,3]
def flatten_and_mask(y_true, y_pred):
  y_true = tf.squeeze(y_true, axis=-1) # remove last dimension to get shape [B,98]
  y_true_flat = tf.reshape(y_true, [-1]) # reshape to [B*98]
  y_pred_flat = tf.reshape(y_pred, [-1, NUM_CLASSES - 1]) #reshape to [B*98, 3] for CORN logits

  # exclude NaN
  valid_mask = tf.math.logical_not(tf.math.is_nan(y_true_flat))
  y_true_valid = tf.boolean_mask(y_true_flat, valid_mask)
  y_pred_valid = tf.boolean_mask(y_pred_flat, valid_mask)

  y_true_valid = tf.cast(y_true_valid, tf.int32)

  return y_true_valid, y_pred_valid


# copied and adapted CornOrdinalCrossEntropy from Github to include class weights and masking:
def weighted_corn_loss(y_true, y_pred):
    
    y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)

    y_true_valid, y_pred_valid = flatten_and_mask(y_true, y_pred) #remove NAN

    weights = tf.gather(CLASS_WEIGHTS, y_true_valid) # inverse freq weights

    sets = []

    for i in range(NUM_CLASSES - 1):
        set_mask = y_true_valid > i - 1
        label_gt_i = y_true_valid > i
        sets.append((set_mask, label_gt_i))

    losses = tf.zeros((), dtype=y_pred.dtype)
    # removed n_examples from original because of graph error

    for task_index, s in enumerate(sets):
        set_mask, label_gt_i = s

        pred_task = tf.gather(y_pred_valid, task_index, axis=1)
        losses_task = tf.where(
            set_mask,
            tf.where(
                label_gt_i,
                tf.math.log_sigmoid(pred_task),
                tf.math.log_sigmoid(pred_task) - pred_task,
            ),
            0.0,
        )

        # Apply class weight BEFORE reducing over samples.
        weighted_losses_task = losses_task * weights

        losses += -tf.reduce_sum(weighted_losses_task)

    return losses / NUM_CLASSES


# Runs the model over every batch in a dataset and sums one confusion matrix across all of them
def accumulate_confusion_matrix(model, dataset):
  confusion_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

  for satellite, label in dataset:
    logits = model(satellite)  # [B, 98, 3]
    true_classes, valid_logits = flatten_and_mask(label, logits)
    predicted_classes = decode_corn_logits(valid_logits)  
    batch_matrix = tf.math.confusion_matrix(
        true_classes, predicted_classes, num_classes=NUM_CLASSES
    ).numpy()
    confusion_matrix += batch_matrix

  return confusion_matrix


# confusion matrix -> metrics (per epoch)
def confusion_matrix_metrics(confusion_matrix, prefix):
  true_positives = np.diag(confusion_matrix) 
  predicted_positives = confusion_matrix.sum(axis=0)  # (column sums)
  actual_positives = confusion_matrix.sum(axis=1)  # (row sums)

  
  precision_per_class = np.divide(
      true_positives, predicted_positives,
      out=np.zeros_like(true_positives, dtype=np.float64), where=predicted_positives > 0,
  )# if pred positives is 0 -> precision is 0

  recall_per_class = np.divide(
      true_positives, actual_positives,
      out=np.zeros_like(true_positives, dtype=np.float64), where=actual_positives > 0,
  )

  f1_per_class = np.divide(
      2 * precision_per_class * recall_per_class,
      precision_per_class + recall_per_class,
      out=np.zeros_like(precision_per_class, dtype=np.float64),
      where=precision_per_class + recall_per_class > 0,
  )

  total_count = confusion_matrix.sum()
  accuracy = float(true_positives.sum() / total_count) if total_count > 0 else 0.0


  class_ids = np.arange(NUM_CLASSES)
  true_class_grid, predicted_class_grid = np.meshgrid(class_ids, class_ids, indexing='ij') 
  absolute_class_errors = np.abs(true_class_grid - predicted_class_grid) # distance matrix for class errors
  mae = float((confusion_matrix * absolute_class_errors).sum() / total_count) if total_count > 0 else 0.0 # sum of all class-distance errors / number of valid labels

  return {
    f'{prefix}_valid_label_count': int(total_count),
    f'{prefix}_ordinal_accuracy': accuracy,
    f'{prefix}_ordinal_mae': mae,
    f'{prefix}_macro_precision': float(precision_per_class.mean()),
    f'{prefix}_macro_recall': float(recall_per_class.mean()),
    f'{prefix}_macro_f1': float(f1_per_class.mean()),
  } #mean because average over all classes

# plot confmatrix (normalized)
def confusion_matrix_to_figure(confusion_matrix, title):
  row_totals = confusion_matrix.sum(axis=1, keepdims=True)

  normalized_matrix = np.divide(
    confusion_matrix,
    row_totals,
    out=np.zeros_like(confusion_matrix, dtype=np.float64),
    where=row_totals != 0,
  )

  fig, ax = plt.subplots(figsize=(6, 5))
  image = ax.imshow(
    normalized_matrix,
    cmap='Blues',
    vmin=0,
    vmax=1,
  )
  fig.colorbar(image, ax=ax, label='Proportion')

  ax.set_title(title)
  ax.set_xlabel('Predicted class')
  ax.set_ylabel('True class')
  ax.set_xticks(np.arange(NUM_CLASSES))
  ax.set_yticks(np.arange(NUM_CLASSES))

  for true_class in range(NUM_CLASSES):
    for predicted_class in range(NUM_CLASSES):
      ax.text(
        predicted_class,
        true_class,
        f'{normalized_matrix[true_class, predicted_class]:.1%}',
        ha='center',
        va='center',
        color='black',
      )

  fig.tight_layout()
  return fig


#creates val set, called after every epoch and calls accumulate_confusion_matrix, logs metrics
class ConfusionMatrixCallback(keras.callbacks.Callback):
  def __init__(self, run):
    super().__init__()
    self.run = run
    self.val_ds = val_data()  # (satellite, label) batches
    self.last_confusion_matrix = None  # final validation result

  def on_epoch_end(self, epoch, logs=None):
    self.last_confusion_matrix = accumulate_confusion_matrix(self.model, self.val_ds)
    figure = confusion_matrix_to_figure(
      self.last_confusion_matrix,
      title=f'Validation confusion matrix - epoch {epoch + 1}',
    )
    self.run.log({'val_confusion_matrix': wandb.Image(figure)}, step=epoch + 1)
    plt.close(figure)
    if logs is not None:
      logs.update(confusion_matrix_metrics(self.last_confusion_matrix, prefix='val')) # pyright: ignore[reportAttributeAccessIssue]




def main():

  model = UNet1D(
      input_shape=(98, 4),
      num_classes=NUM_CLASSES - 1,
      final_activation="linear",
      filters=64
  )
  # 4 classes require 3 ordinal thresholds
  # corn requires raw logits, therefore no sigmoid activation function

  wandb.init(project='Corals')
  run = cast(Any, wandb.run)
  print(f'models/{run.id}.keras')

  model.compile(
      optimizer=Adam(),
      loss=weighted_corn_loss
  )
  
  confusion_callback = ConfusionMatrixCallback(run)
  
  # Fit with only the callback for validation
  model.fit(
      train_data(),
      epochs=20,
      validation_data=val_data(),
      callbacks=[confusion_callback, WandbMetricsLogger()],
  )
  model.save(f'models/{run.id}.keras')

  # use test set for final evaluation if available, otherwise use last validation matrix
  if test_data is not None:
    final_confusion_matrix = accumulate_confusion_matrix(model, test_data())
    final_prefix = 'test'
  else:
    final_confusion_matrix = confusion_callback.last_confusion_matrix
    final_prefix = 'val_final'
  
  final_metrics = confusion_matrix_metrics(final_confusion_matrix, prefix=final_prefix)
  print(f'Final metrics ({final_prefix}):')
  print(final_metrics)
  run.summary.update(final_metrics)

  final_confusion_figure = confusion_matrix_to_figure(
    final_confusion_matrix,
    title=f'Final confusion matrix - {final_prefix}',
  )
  run.log({'final_confusion_matrix': wandb.Image(final_confusion_figure)})
  plt.close(final_confusion_figure)




if __name__ == "__main__": 
  main()
