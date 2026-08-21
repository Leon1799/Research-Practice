import cubo
import numpy as np
import pandas as pd
from dask.diagnostics.progress import ProgressBar
import ee
import xee
import geemap
import rioxarray
import xarray as xr
from tqdm.notebook import tqdm
import h5netcdf
#import pypeln.thread as pl
from pathlib import Path
import pystac_client
import planetary_computer

catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)

ee.Initialize(project='glassy-acolyte-480612-n3')
ACA = ee.Image("ACA/reef_habitat/v2_0")
ls8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
    .select(['SR_B2', 'SR_B3', 'SR_B4', 'ST_B10', 'QA_PIXEL']) \
    .map(lambda img: img.rename(['blue', 'green', 'red', 'lwir', 'qa_pixel']))
ls9 = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2') \
    .select(['SR_B2', 'SR_B3', 'SR_B4', 'ST_B10', 'QA_PIXEL']) \
    .map(lambda img: img.rename(['blue', 'green', 'red', 'lwir', 'qa_pixel']))
ls7 = ee.ImageCollection('LANDSAT/LE07/C02/T1_L2') \
    .select(['SR_B1', 'SR_B2', 'SR_B3', 'ST_B6', 'QA_PIXEL']) \
    .map(lambda img: img.rename(['blue', 'green', 'red', 'lwir', 'qa_pixel']))
landsat = ls8.merge(ls9).merge(ls7)

ROOT = Path('./data')
(ROOT / 'ACA').mkdir(exist_ok=True, parents=True)
(ROOT / 'nc').mkdir(exist_ok=True, parents=True)

# checks if site already downloaded
def download(site):
  print(f'{site.region_name}: Starting download')
  out_path = ROOT / 'nc' / f'{site.region_name}.nc'
  if out_path.exists():
    print(f'{site.region_name}: Skipping')
    return


  da = cubo.create(
    lat=site.lat, # Central latitude of the cube
    lon=site.lon, # Central longitude of the cube
    collection=landsat, # Name of the STAC collection
    bands=['blue', 'green', 'red', 'lwir', 'qa_pixel'], # Bands to retrieve
    start_date="2000-01-01", # Start date of the cube
    end_date="2024-12-31", # End date of the cube
    edge_size=256, # Edge size of the cube (px)
    resolution=30, # Pixel size of the cube (m)
    stackstac_kw=dict(
      dtype=np.uint16,
      rescale=False,
      fill_value=0,
    ),
    gee=True, # Set to True
    query={"eo:cloud_cover": { "lt": 10 }} # kwarg to pass
  )

  da = da.fillna(0).astype(np.uint16)
  ds = da.to_dataset(dim='band')
  del ds.attrs['collection']

  print(f'{site.region_name}: Valid data filtering')
  # Filter valid acquisitions
  bitmask = (1 << 3) | (1 << 1) | (1 << 4)
  valid_mask = (ds.qa_pixel & bitmask) == 0
  valid_times = valid_mask.mean(dim=['y', 'x']) > 0.1
  with ProgressBar():
    valid_times = valid_times.compute()
  ds = ds.isel(time=valid_times)

  dx = float(ds.x[1] - ds.x[0])
  dy = float(ds.y[1] - ds.y[0])
  tx = float(ds.x[0])
  ty = float(ds.y[0])

  transform = [
      dx, 0, tx - dx/2,
      0, dy, ty - dy/2,
      ]
  # import pdb; pdb.set_trace();
  crs = 'EPSG:' + str(int(ds.epsg))

  print(f'{site.region_name}: ACA Download')
  aca_path = ROOT / 'ACA' / f'ACA_{site.region_name}.tif'
  geemap.download_ee_image(
    ACA,
    aca_path,
    crs=crs,
    crs_transform=transform,
    shape=[len(ds.x), len(ds.y)],
    dtype='uint8'
  )

  aca = xr.open_dataset(aca_path, decode_coords="all", engine="rasterio")
  aca = aca.fillna(255).astype(np.uint8)
  aca = aca.band_data.to_dataset(dim='band')
  aca = aca.rename_vars({1: 'geomorphic', 2: 'benthic', 3: 'reef_mask'})

  print(f'{site.region_name}: Merging Datasets')
  assert np.all(aca.x.values == ds.x.values)
  assert np.all(aca.y.values == ds.y.values)
  ds = ds.merge(aca)

  print(f'{site.region_name}: Saving...')
  for var in ds.variables:
    ds[var].encoding = { 'zlib': True, 'complevel': 3, 'shuffle': True }

  tmp_path = out_path.with_suffix('.partial.nc')
  with ProgressBar():
    ds.to_netcdf(tmp_path, engine='h5netcdf')
  tmp_path.rename(out_path)
  print(f'{site.region_name}: Done.')


def main():
  import sys
  _, offset, mod = sys.argv
  regions = pd.read_parquet(r'./Scripts/Labels/regions.parquet')########################
  regions['region_name'] = regions.Site_ID.apply(lambda x: f'SID_{x}')
  regions = [r for r in regions.itertuples() if not (ROOT / 'nc' / f'{r.region_name}.nc').exists()]
  regions = regions[int(offset)::int(mod)]

  for region in tqdm(regions):
    try:
      download(region)
    except Exception as e:
      print(f'Error for {region.region_name}')
      print(e)

if __name__ == "__main__":
  main()
