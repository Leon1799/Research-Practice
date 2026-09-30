"""corals dataset."""

from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import xarray as xr

import tensorflow as tf
import tensorflow_datasets.public_api as tfds


# Set to True to use the site-level spatial split from splits.parquet (train/val/test).
# Keep False for fast iteration on the reduced 12-site labels.parquet (train/val only).
USE_SPATIAL_SPLIT = True


class Builder(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for corals dataset."""

  VERSION = tfds.core.Version('3.0.2')
  RELEASE_NOTES = {
      '1.0.0': 'Local, 12 sites with bleaching threshold',
      '1.0.1': 'Server, 12 sites with bleaching threshold',
      '1.0.2': 'Local, 12 sites with bleaching threshold, spatial splitting',
      '1.0.3': 'Server, spatial splitting, 500 sites',
      '1.0.4': 'Server, all Indo-Pacific sites',
      '1.0.5': 'Server, all Indo-Pacific sites without 1 labeled quarter sites',
      '2.0.0': 'Local, 12 sites with no threshold and new class labels',
      '2.0.1': 'Server, all Indo-Pacific sites, class labels',
      '2.0.2': 'Server, all Indo-Pacific sites, spatial split balanced on 4 classes',
      '2.0.3': 'Server, all sites, spatial split balanced on 4 classes',
      '2.0.4': 'Server, all sites, "", undersampled train set',
      '3.0.0': 'Local, 12 sites, sentinel test',
      '3.0.1': 'Local, 12 sites, sentinel test, cloud probability, outlier threshold',
      '3.0.2': 'Server, all sites',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(corals): Specifies the tfds.core.DatasetInfo object
    return self.dataset_info_from_configs(
        features=tfds.features.FeaturesDict({
            'satellite': tfds.features.Tensor(shape=(38, 4), dtype=np.float32, encoding=tfds.features.Encoding.BYTES),
            'label': tfds.features.Tensor(shape=(38, 1), dtype=np.float32, encoding=tfds.features.Encoding.BYTES),
            'site': tfds.features.Text(),
        }),
        supervised_keys=('satellite', 'label'),  # Set to `None` to disable
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    tfds.disable_progress_bar()

# if running with reduced 12 site dataset testlabels.parquet (with simple SID_ID splitting) is used and if running on whole dataset labels.parquet (with site level spatial splitting) is used
    if USE_SPATIAL_SPLIT:
      labels = pd.read_parquet(r'./Scripts/Labels/labels.parquet')
      return self._split_generators_spatial(labels)

    labels = pd.read_parquet(r'./Scripts/Labels/sentinellabels.parquet')
    train = labels[~labels.index.str.endswith('8')]
    val = labels[labels.index.str.endswith('8')]

    print(f'Train: {len(train)}')
    print(f'Val: {len(val)}')

    # TODO(corals): Returns the Dict[split names, Iterator[Key, Example]]
    return {
        'train': self._generate_examples(train),
        'val': self._generate_examples(val),
    }

  def _split_generators_spatial(self, labels):
    """Returns train, validation, and test splits by site."""
    splits = pd.read_parquet(
        r'./Scripts/Labels/splits.parquet'
    ).set_index('sid')

    site_split = splits['split'].reindex(labels.index)

    train = labels[site_split == 'train']
    val = labels[site_split == 'val']
    test = labels[site_split == 'test']

    print(f'Train: {len(train)}')
    print(f'Val: {len(val)}')
    print(f'Test: {len(test)}')

    return {
        'train': self._generate_examples(train),
        'val': self._generate_examples(val),
        'test': self._generate_examples(test),
    }
  
  def _generate_examples(self, labels):
    """Yields examples."""
    timeseries = {}
    for site, label in tqdm(list(labels.iterrows())):
      ncpath = Path(r'./data/timeseries') / f'{site}.nc'
      if not ncpath.exists(): # skip if timeseries file doesn´t exists because no coral e.g
        continue
      ts = xr.open_dataarray(ncpath, engine='h5netcdf')

      for idx, pix in ts.groupby('pixel', squeeze=False):
        satellite = pix.squeeze('pixel').sel(band=['blue', 'green', 'red', 'nir']).transpose('time', 'band') / 10_000
        satellite = satellite.values.astype(np.float32)
        yield f'{site}_{idx}', {
          'satellite': satellite,
          'label': label.values.astype(np.float32).reshape((38, 1)),
          'site': site,
        }
