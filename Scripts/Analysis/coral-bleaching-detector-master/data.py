import tensorflow as tf
import tensorflow_datasets as tfds
from typing import cast
from pathlib import Path
import numpy as np
import pandas as pd

MAX_DATA_WORKERS = 3 # limit due to server limitations

NUM_CLASSES = 4

def train_data():
  ds = cast(
    tf.data.Dataset,
    tfds.load('corals', split='train', shuffle_files=True)
  )
  ds = ds.shuffle(10_000) 
  ds = ds.batch(1024) #batchsize
  ds = ds.map(prepare_sample, num_parallel_calls=MAX_DATA_WORKERS)
  ds = ds.map(augment, num_parallel_calls=MAX_DATA_WORKERS)
  return ds


def val_data(yield_dicts=False):
  ds = cast(
    tf.data.Dataset,
    tfds.load('corals', split='val')
  )
  ds = ds.batch(1024) 
  if yield_dicts:
    ds = ds.map(prepare_sample_dicts)
  else:
    ds = ds.map(prepare_sample)
  return ds


# if running code with reduced 12 site dataset (during code development) no test split is available
if 'test' in tfds.builder('corals').info.splits:
  def test_data(yield_dicts=False):
    ds = cast(
      tf.data.Dataset,
      tfds.load('corals', split='test')
    )
    ds = ds.batch(1024)
    if yield_dicts:
      ds = ds.map(prepare_sample_dicts)
    else:
      ds = ds.map(prepare_sample)
    return ds
else:
  globals().pop('test_data', None)



def augment(timeseries, label):
  #t0 = tf.random.uniform([], 0, 98-66, dtype=tf.int32)
  #timeseries = timeseries[:, t0:t0+66]
  #label = label[:, t0:t0+66]
  return timeseries, label


def prepare_sample(sample):
  ts = sample['satellite']
  label = sample['label']
  return ts, label


def prepare_sample_dicts(sample):
  sample = dict(**sample)
  return sample



# inverse frequency class weights
def class_weights():
    train_dataset = tfds.load("corals", split="train")
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)

    for sample in tfds.as_numpy(train_dataset):
        values = np.asarray(sample["label"]).reshape(-1) # convert label to 1dim np array (98,)

        #drop NAN quarters
        values = values[np.isfinite(values)]

        values = values.astype(np.int64)

        # Count occurrences of classes 
        counts += np.bincount(
            values,
            minlength=NUM_CLASSES,
        )

    present = counts > 0

    weights = np.ones(NUM_CLASSES, dtype=np.float32)

    weights[present] = (
        counts[present].sum()
        / (present.sum() * counts[present])
    )

    print("Training class counts:", counts)
    print("Training class weights:", weights)

    return weights

