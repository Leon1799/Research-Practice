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

S2_CLOUD_PROBABILITY_THRESHOLD = 30 #try out best threshold in Plotting_Canaries

# mask SCL in gee
def mask_s2(image):
    scl = image.select('SCL')
    cloud_probability = ee.Image(image.get('cloud_probability')).select('probability')

    valid = (
        scl.neq(1) # Saturated or defective
        .And(scl.neq(2)) # Dark Area Pixels
        .And(scl.neq(3)) # Cloud Shadows
        .And(scl.neq(7)) # Clouds Low Probability 
        .And(scl.neq(8)) # Clouds Medium Probability
        .And(scl.neq(9)) # Clouds High Probability
        .And(scl.neq(10)) # Cirrus
        .And(scl.neq(11)) # Snow / Ice
    )

    valid = valid.And(cloud_probability.lt(S2_CLOUD_PROBABILITY_THRESHOLD))

    return image.updateMask(valid)

ee.Initialize(project='glassy-acolyte-480612-n3')
ACA = ee.Image("ACA/reef_habitat/v2_0")

sentinel = (
    ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterDate('2017-04-01', '2026-08-01')
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 90)) # filter images with >90% cloud cover
)

cloud_probability = (
    ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
    .filterDate('2017-04-01', '2026-08-01')
)

joined = ee.Join.saveFirst('cloud_probability').apply(
    primary=sentinel,
    secondary=cloud_probability,
    condition=ee.Filter.equals(
        leftField='system:index',
        rightField='system:index',
    ),
)

sentinel = (
    ee.ImageCollection(joined)
    .filter(ee.Filter.notNull(['cloud_probability']))
    .map(mask_s2)
    .select(['B2', 'B3', 'B4', 'B8'])
    .map(lambda img: img.rename(['blue', 'green', 'red', 'nir']))
)

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
    collection=sentinel, # Name of the STAC collection
    bands=['blue', 'green', 'red', 'nir'], # Bands to retrieve
    start_date="2017-04-01", # Start date of the cube
    end_date="2026-08-01", # End date of the cube
    edge_size=256, # Edge size of the cube (px)
    resolution=10, # Pixel size of the cube (m)
    stackstac_kw=dict(
      dtype=np.uint16,
      rescale=False,
      fill_value=0, 
    ),
    gee=True, # Set to True
#    query={"eo:cloud_cover": { "lt": 10 }} # kwarg to pass
  )

  da = da.fillna(0).astype(np.uint16) 
  ds = da.to_dataset(dim='band')
  del ds.attrs['collection']

  print(f'{site.region_name}: Valid data filtering')


# sentinel pixel coordinates
  dx = float(ds.x[1] - ds.x[0])
  dy = float(ds.y[1] - ds.y[0])
  tx = float(ds.x[0])
  ty = float(ds.y[0])

# ensure that ACA has same pixel grid as sentinel (10m)
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
  assert np.all(aca.x.values == ds.x.values) # verify that aca matches sentinel
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

# change to testregions.parquet if local code development
def main():
  import sys
  _, offset, mod = sys.argv
  regions = pd.read_parquet(r'./Scripts/Labels/regions.parquet')
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
