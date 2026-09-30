import tensorflow as tf
import tensorflow_datasets as tfds
from typing import cast
from pathlib import Path
import numpy as np
import pandas as pd

MAX_DATA_WORKERS = 3 # limit due to server limitations

NUM_CLASSES = 4





# site aware undersampling
#PROJECT_ROOT = Path(__file__).resolve().parents[3] 
#LABELS_PATH = PROJECT_ROOT / "Scripts" / "Labels" / "labels.parquet"

# compute the ids that are removed
#def removable_site_ids():
#  labels = pd.read_parquet(LABELS_PATH)

  # Remove sites that only contain majority class 0 or NaN
#  pure_class_0 = labels.fillna(0).eq(0).all(axis=1)
 # removed_site_ids = labels.index[pure_class_0]

#  print(f"Undersampling removed {len(removed_site_ids)} class-0/NaN-only sites.")
#  return tf.constant(removed_site_ids.to_numpy(), dtype=tf.string)

# apply those ids
#def undersample_train_sites(dataset):
  # Look up table for removable site IDs because tfds gives examples one by one
#  removed_site_ids = removable_site_ids()
#  removed_site_lookup = tf.lookup.StaticHashTable(
#    tf.lookup.KeyValueTensorInitializer(
#      removed_site_ids,
#      tf.ones(tf.shape(removed_site_ids), dtype=tf.int32),
#    ),
#    default_value=0,
#  )

  # Keep every coral-pixel example from retained sites and drop all removable sites
#  return dataset.filter(lambda sample: removed_site_lookup.lookup(sample["site"]) == 0)





# seed so that examples appear in same shuffled order
def train_data(seed):
  ds = cast(
    tf.data.Dataset,
    tfds.load('corals', split='train', shuffle_files=True, read_config=tfds.ReadConfig(shuffle_seed=seed))
  )
#  ds = undersample_train_sites(ds)
  ds = ds.shuffle(10_000, seed=seed, reshuffle_each_iteration=True) 
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
  #t0 = tf.random.uniform([], 0, 38-66, dtype=tf.int32)
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



# inverse frequency (sqrt) class weights
def class_weights():
    train_dataset = tfds.load("corals", split="train")
#    train_dataset = undersample_train_sites(train_dataset)
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)

    for sample in tfds.as_numpy(train_dataset):
        values = np.asarray(sample["label"]).reshape(-1) # convert label to 1dim np array (38,)

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

    inverse_sqrt = 1.0 / np.sqrt(counts[present].astype(np.float32))
    weights[present] = inverse_sqrt / inverse_sqrt.mean()

    print("Training class counts:", counts)
    print("Training class weights:", weights)

    return weights

