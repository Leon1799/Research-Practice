import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras # pyright: ignore[reportMissingModuleSource]
from train import bce, accuracy_at_10
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from data import val_data # pyright: ignore[reportAttributeAccessIssue]
from sys import argv

# plots validation sites
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
    labels.append(batch['label'].numpy())

  sites = np.concatenate(sites)
  labels = np.concatenate(labels)
  preds = np.concatenate(preds)

  # for local code development change path: 'viz'
  out_dir = Path('/home/leon/viz') / MODEL 
  out_dir.mkdir(parents=True, exist_ok=True)

  t = pd.date_range('2000-01-01', '2024-04-01', freq='QS')
  for site in tqdm(np.unique(sites)):
    mask = sites == site
    site = site.decode()

    fig, ax = plt.subplots(figsize=(7, 5))
    plt.title(site)
    ax.plot(t, 100 * np.nanmean(preds[mask], axis=0), c='b')
    ax.plot(t, 100 * np.nanmax(preds[mask], axis=0), alpha=0.2, c='b')
    ax.plot(t, 100 * np.nanmin(preds[mask], axis=0), alpha=0.2, c='b')
    ax.scatter(t, 100 * labels[mask][0], marker='o', c='r')
    # ax.set_ylim(0, 100)
    plt.savefig(out_dir / f'{site}.png')
    plt.close(fig)

  # Full aggregate plot
  fig, ax = plt.subplots(figsize=(7, 5))
  plt.title('All Sites')
  ax.plot(t, 100 * np.nanmean(preds, axis=0), c='b')
  ax.plot(t, 100 * np.nanmax(preds, axis=0), alpha=0.2, c='b')
  ax.plot(t, 100 * np.nanmin(preds, axis=0), alpha=0.2, c='b')
  # ax.set_ylim(0, 100)
  plt.savefig(out_dir / f'all.png')
  plt.close(fig)



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
