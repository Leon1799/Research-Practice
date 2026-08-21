"""corals dataset."""

from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import xarray as xr

import tensorflow as tf
import tensorflow_datasets.public_api as tfds


class Builder(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for corals dataset."""

  VERSION = tfds.core.Version('1.0.1')
  RELEASE_NOTES = {
      '1.0.0': 'gentle-serenity-17 Local Test (12 sites)',
      '1.0.1': '? Server Test (12 sites)',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(corals): Specifies the tfds.core.DatasetInfo object
    return self.dataset_info_from_configs(
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'satellite': tfds.features.Tensor(shape=(98, 4), dtype=np.float32, encoding=tfds.features.Encoding.BYTES),
            'label': tfds.features.Tensor(shape=(98, 1), dtype=np.float32, encoding=tfds.features.Encoding.BYTES),
            'site': tfds.features.Text(),
        }),
        supervised_keys=('satellite', 'label'),  # Set to `None` to disable
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    labels = pd.read_parquet(r'./Scripts/Labels/labels.parquet')
    tfds.disable_progress_bar()

##################
# changed from '3' to 'f' to match the new label file
    train = labels[~labels.index.str.endswith('f')]
    val = labels[labels.index.str.endswith('f')]
################

    print(f'Train: {len(train)}')
    print(f'Val: {len(val)}')

    # TODO(corals): Returns the Dict[split names, Iterator[Key, Example]]
    return {
        'train': self._generate_examples(train),
        'val': self._generate_examples(val),
    }

  def _generate_examples(self, labels):
    """Yields examples."""
    timeseries = {}
    for site, label in tqdm(list(labels.iterrows())):
      ncpath = Path(r'./data/timeseries') / f'{site}.nc'###########################################
      if not ncpath.exists(): # skip if timeseries file doesn´t exists because no coral e.g
        continue
      ts = xr.open_dataarray(ncpath, engine='h5netcdf')

      for idx, pix in ts.groupby('pixel', squeeze=False):
        satellite = pix.squeeze('pixel').sel(band=['blue', 'green', 'red', 'lwir']).transpose('time', 'band') / 10_000
        satellite = satellite.values.astype(np.float32)
        yield f'{site}_{idx}', {
          'satellite': satellite,
          'label': label.values.astype(np.float32).reshape((98, 1)),
          'site': site,
        }
