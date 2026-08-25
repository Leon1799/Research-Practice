import tensorflow as tf
import tensorflow_datasets as tfds
from typing import cast

MAX_DATA_WORKERS = 3 # limit due to server limitations

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
  t0 = tf.random.uniform([], 0, 98-66, dtype=tf.int32)
  timeseries = timeseries[:, t0:t0+66]
  label = label[:, t0:t0+66]
  # Add random gaussian noise
  # timeseries = timeseries + 0.2 * tf.random.normal(tf.shape(timeseries))
  return timeseries, label


def prepare_sample(sample):
  ts = sample['satellite']
  # Assumption 1: All labels are given in percent, so divide them by 100 to get [0-1] values
  label = sample['label'] / 100
  # Alternative (Bold) Assumption 2: All values below 1 are meant as fractions while
  # values above 1 are given as percent values
  # label = tf.where(label > 1, label / 100, label) # Scale % to [0-1]
  return ts, label


def prepare_sample_dicts(sample):
  sample = dict(**sample)
  sample['label'] = sample['label'] / 100
  # Alternative (Bold) Assumption 2: All values below 1 are meant as fractions while
  # values above 1 are given as percent values
  # label = tf.where(label > 1, label / 100, label) # Scale % to [0-1]
  return sample

