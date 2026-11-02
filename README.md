# Estimating Coral Bleaching from Multispectral Satellite Time Series using Deep Learning

This research project investigates whether satellite time series can be used to estimate coral bleaching severity across reef sites. It combines in-situ observations from Reef Check, coral habitat masks from the Allen Coral Atlas, and satellite imagery to train a 1D U-Net.

The final experiment uses Sentinel-2 imagery and **ordinal classification** to predict four ordinal bleaching severity classes. The research also explored Landsat imagery, a regression baseline, spatial data splitting, class weighting, and training-set undersampling.

## Project summary

The pipeline links ReefCheck population-bleaching observations to coral pixels, processes satellite measurements into quarterly time series, and trains a model to estimate bleaching severity for each pixel and quarter:

1. Select reef sites and create quarterly labels and spatial train/validation/test splits.
2. Download Sentinel-2 imagery and aligned Allen Coral Atlas habitat masks for each site.
3. Filter clouds and bright outliers, select coral pixels, and compute quarterly median reflectance.
4. Build a TensorFlow dataset and train a 1D U-Net with CORN ordinal classification.

The final configuration uses a class-balanced spatial split and square-root inverse-frequency class weights. The model takes 38 quarterly observations from 2017 to 2026 with four bands (blue, green, red, and near-infrared) per coral pixel.


## Data

| Source | Use |
| --- | --- |
| [Reef Check Global Reef Dataset](https://data.reefcheck.org/) | In-situ population-bleaching observations used to create site-level labels. |
| [Sentinel-2 surface reflectance](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED) | Blue, green, red, and near-infrared imagery for the final experiment, accessed through Google Earth Engine. |
| [Allen Coral Atlas](https://developers.google.com/earth-engine/datasets/catalog/ACA_reef_habitat_v2_0) | Benthic habitat masks used to select coral pixels. |

The report also compares earlier experiments using Landsat imagery; the current downloader is configured for Sentinel-2. Reef Check observations are averaged across survey segments and across observations within a quarter. Each coral pixel at a site receives the same site-level quarterly label.

The current label conversion uses these boundaries:

| Class | Severity | Population bleaching |
| --- | --- | --- |
| 0 | None | < 1% |
| 1 | Low | 1% to 10% |
| 2 | Moderate | 10% to 50% |
| 3 | Severe | > 50% |

The repository includes the generated `regions.parquet`, `labels.parquet`, and `splits.parquet` files. The original Reef Check `Belt.csv`, satellite cubes, extracted time series, trained models, and experiment logs are not included. 

## Repository structure

```text
Scripts/
  Labels/
    siteselection and data viz.ipynb  Site selection, labels, and spatial splits
    regions.parquet                   Selected site coordinates
    labels.parquet                    Quarterly severity labels
    splits.parquet                    Site-level train/validation/test split
  Downloader/
    main.py                          Sentinel-2 and ACA-mask download
  Analysis/coral-bleaching-detector-master/
    datasets/corals/
      extract_ts_SLnotes.py          Coral-pixel time-series extraction
      corals_dataset_builder.py      TensorFlow Datasets builder
    data.py                          Dataset loading and class weights
    unet1d.py                        1D U-Net architecture
    train.py                         Training and evaluation
Dockerfile                           GPU container configuration
requirements_311.txt                 Pinned Python dependencies
```

Generated files are stored under `data/`, `models/`, and `wandb/` and are excluded from Git.

## Installation

The latest experiment uses **Python 3.11** and **TensorFlow 2.21.0**. Site selection can be run locally; downloading and training the full dataset require substantial storage, time, and GPU resources. The reported experiments used an NVIDIA GeForce RTX 3090.

## Running the pipeline

Run the following commands **from the repository root**. The settings below describe the latest experiment; they can be changed for other experiments.

### 1. Select sites and prepare labels

Open [`Scripts/Labels/siteselection and data viz.ipynb`]. Obtain the Reef Check `Belt.csv` separately and place it under `Scripts/Labels/`.


The latest configuration is:

```python
N_SITES = "all"
REGION = "all"
SEVERE_BLEACHING_THRESHOLD = 0
MIN_LABEL_QUARTERS = 0
START_QUARTER = "2017-04-01"
END_QUARTER = "2026-08-01"
```

For a smaller local experiment, set `N_SITES` to an integer, such as `12`, and regenerate all three reduced Parquet files.

### 2. Configure the dataset and credentials

1. For a new dataset, update both `VERSION` and `RELEASE_NOTES` in [`corals_dataset_builder.py`].
2. Check that [`main.py`] reads `Scripts/Labels/regions.parquet`. If using a different reduced regions file, change the input path there.
3. Set `USE_SPATIAL_SPLIT = True` in `corals_dataset_builder.py` for the full spatial train/validation/test workflow. Set `False` to read a reduced labels file.
4. In `main.py`, replace the hard-coded Google Earth Engine project in `ee.Initialize(project=...)`


### 3. Download satellite data and extract time series

```bash
python Scripts/Downloader/main.py 0 1
python Scripts/Analysis/coral-bleaching-detector-master/datasets/corals/extract_ts_SLnotes.py
```

 The downloader saves Sentinel-2 cubes in `data/nc/` and masks in `data/ACA/`.

### 4. Train and evaluate

```bash
python Scripts/Analysis/coral-bleaching-detector-master/train.py
```

To compare against training without class weights:

```bash
python Scripts/Analysis/coral-bleaching-detector-master/train.py --no-class-weights
```

The current configuration trains for 20 epochs and saves the best checkpoint, selected by validation macro F1, as `models/<wandb-run-id>.keras`. It evaluates that checkpoint on the test split when available. Metrics and confusion matrices are logged to Weights & Biases.
