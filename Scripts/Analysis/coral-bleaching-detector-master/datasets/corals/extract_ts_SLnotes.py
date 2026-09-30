import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from tqdm import tqdm
import pypeln.process as pl
import shutil

def extract(in_path):
  out_path = in_path.parent.parent / 'timeseries' / in_path.name
  ds = xr.open_dataset(in_path)
  data = ds.stack(pixel=['y', 'x'])
  coral = data.sel(pixel=(data.benthic == 15)) # select coral pixels
  # reef = data.sel(pixel=(data.reef_mask == 1))
  if len(coral.pixel) == 0:
    print(f'No coral pixels for {in_path.stem}!') # no coral pixels if reefcheck data label exists but not categorized in ACA -> then no timeseries will be created for that site
    return


  # outliers are clouds, whitecaps
  outlier = (
    (coral["red"] > 1000)
    & (coral["green"] > 1000)
    & (coral["blue"] > 1000)
    & (coral["nir"] > 1000)
  )

  for channel in ['blue', 'red', 'green', 'nir']:
    coral[channel] = coral[channel].where((coral[channel] > 0) & (~outlier)) 

  masked = coral
  resampled = masked.sortby('time').resample(time='QS', skipna=True).median()

  resampled = resampled.reset_index('pixel')

  # Reindex timeseries 
  new_times = pd.date_range(start='2017-04-01', end='2026-07-01', freq='QS')
  resampled = resampled.reindex(time=new_times)

  # Interpolate missing data points in-between valid data points
  #resampled = resampled.interpolate_na('time', 'linear') ### linear by default take mean of two adjacent time points###
  # Extrapolate missing data outside the acquisition period
  # resampled = resampled.interpolate_na('time', 'nearest', fill_value='extrapolate')
  resampled = resampled.fillna(0) ###fill bookend time points with 0. alternative would be line above.###

  for var in resampled.variables:
    resampled[var].encoding = { 'zlib': True, 'complevel': 3, 'shuffle': True }

  resampled[['blue', 'green', 'red', 'nir']].to_dataarray('band').to_netcdf(out_path)



if __name__ == "__main__":
    timeseries_dir = Path("data/timeseries")

    if timeseries_dir.exists():
        shutil.rmtree(timeseries_dir)

    timeseries_dir.mkdir(parents=True, exist_ok=True)

    paths = list(Path("data/nc").glob("*.nc"))
    pipeline = pl.map(
        extract, # pyright: ignore[reportArgumentType]
        paths,
        workers=3, 
    )  

    for _ in tqdm(pipeline, total=len(paths), ncols=80):
        pass