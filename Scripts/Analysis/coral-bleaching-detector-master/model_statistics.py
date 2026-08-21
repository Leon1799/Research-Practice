import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras # pyright: ignore[reportMissingModuleSource]
from train import bce, accuracy_at_10
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sys import argv

from data import val_data # pyright: ignore[reportAttributeAccessIssue]

MODEL = argv[1]

# for local code development change path: f'models/{MODEL}.keras'
def main():
  ds = val_data(yield_dicts=True)
  
  model = keras.models.load_model(
      f'/home/leon/models/{MODEL}.keras',
      custom_objects={'bce': bce, 'accuracy_at_10': accuracy_at_10}
  )


  preds = []
  sites = []
  labels = []

  for batch in tqdm(ds):
    sites.append(batch['site'].numpy())
    preds.append(model(batch['satellite']).numpy()[..., 0])
    preds[-1][(batch['satellite'] == 0).numpy().all(axis=-1)] = np.nan
    labels.append(batch['label'].numpy()[..., 0])

  sites = np.concatenate(sites)
  labels = np.concatenate(labels)
  preds = np.concatenate(preds)

  xs = []
  ys = []

  for site in tqdm(np.unique(sites)):
    mask = sites == site
    site = site.decode()

    label = labels[mask][0]
    predmean = np.nanmean(preds[mask], axis=0)
    nanmask = ~(np.isnan(label) | np.isnan(predmean))
    xs.append(label[nanmask])
    ys.append(predmean[nanmask])

  xs = np.concatenate(xs)
  ys = np.concatenate(ys)
  print('xs nan', 100 * np.mean(np.isnan(xs)), '%')
  print('ys nan', 100 * np.mean(np.isnan(ys)), '%')

  # https://en.wikipedia.org/wiki/Coefficient_of_determination
  res = ys - xs
  ss_res = np.sum(np.square(res))
  ss_tot = np.sum(np.square(xs - xs.mean()))

  print('R²:', 1 - ss_res / ss_tot)


def fix_labels(sample):
  # Assumption 1: All labels are given in percent, so divide them by 100 to get [0-1] values
  sample = dict(**sample)
  sample['label'] = sample['label'] / 100
  # Alternative (Bold) Assumption 2: All values below 1 are meant as fractions while
  # values above 1 are given as percent values
  # label = tf.where(label > 1, label / 100, label) # Scale % to [0-1]
  return sample


if __name__ == "__main__":
  main()
