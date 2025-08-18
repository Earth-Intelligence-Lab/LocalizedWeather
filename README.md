# Localized Weather

This repository contains the code and data for the paper ["Local Off-Grid Weather Forecasting with Multi-Modal Earth Observation Data"](https://arxiv.org/abs/2410.12938).

The paper presents a novel multi-modal method that downscales gridded weather forecasts, such as ERA5 and HRRR, to provide accurate off-grid predictions. The model leverages both global data and local weather station observations from MADIS to make predictions that reflect both large-scale atmospheric dynamics and local weather patterns.

The model is evaluated on a surface wind prediction task and shows significant improvement over baseline methods, including ERA5 and HRRR interpolation and a multi-layer perceptron.

Use the following citation when these data or model are used:
> Yang, Q.; Giezendanner, J.; Civitarese, D. S.; Jakubik, J.; 
,Schmitt E.; Chandra, A.; Vila, J.; Hohl, D.; Hill, C.; Watson, C.; Wang, S.; Local Off-Grid Weather Forecasting with Multi-Modal Earth Observation Data. arXiv, October 2024. https://doi.org/10.48550/arXiv.2410.12938



# Model and data
## Data

The data for training, testing and validation can be found on [Zenodo](https://zenodo.org/records/15346612).

The following data is available:
- Shapefile of the Northeastern United States (NE-US, extracted from [NWS](https://www.weather.gov/gis/USStates))
- Shapefile containing the location and number of observations (2019-2023) of the MADIS stations in NE-US
- Processed hourly averaged [MADIS](https://madis.ncep.noaa.gov/) data for the NE-US (2019-2023)
- [ERA5](https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation) data for the NE-US (2019-2023), gridded and interpolated
- HRRR tbd

For MADIS, ERA5 and HRRR, the following variables are available:
- u and v component of wind vector at 10 meters above ground
- temperature at 2 meters above ground
- dewpoint at 2 meters above ground

The data can also be generated from scratch.
For this, code is available under `Source/DataDownload/{ERA5/MADIS/HRRR}` for the raw data download, and in the respective data loaders for the data processing (`Source/Dataloader/{ERA5/HRRR/MetaStation/Madis/MixDataMLP}.py`).

## Code
The code is organised as follows (in `Source/`):
- `LocalLauncher.py`, and `SLURM_job_launcher.py` are two different launchers all eventually pointing at `Main.py` (the arguments from the slurm job launcher are parsed by `Arg_Parser.py`)
- `Main.py` contains the main code loop
- `EvaluateModel` contains the code for the evaluation of the model, as well as the back propagation
- the folder `Dataloader/` contains the data loaders for MADIS, HRRR and ERA5, and the combination of them, and the folder `Network/` the code for the network construction, for both the internal (MADIS) and external (ERA5 to MADIS) connections
- `Modules/GNN/MPNN.py` contains the code for the heterogeneous message passing neural network and calls `GNN_Layer_Internal/External.py`, the message passing sequences between the networks
- `Modules/Transformer/ViT.py` contains the code for the transformer and calls `Transformer/StationsEmbedding.py`.

### Code inputs
The code expects the following data structure, you need to specify the root data path in the launcher (to be updated with HRRR data):
```
RootDataPath/
├── ERA5
│   ├── Interpolated
│   │   ├── era5interpolated_e2m_8_2019_-80.53_-66.94_38.92_47.47northeastern_buffered_filtered_0.9.nc
│   │   ├── era5interpolated_e2m_8_2020_-80.53_-66.94_38.92_47.47northeastern_buffered_filtered_0.9.nc
│   │   ├── era5interpolated_e2m_8_2021_-80.53_-66.94_38.92_47.47northeastern_buffered_filtered_0.9.nc
│   │   ├── era5interpolated_e2m_8_2022_-80.53_-66.94_38.92_47.47northeastern_buffered_filtered_0.9.nc
│   │   └── era5interpolated_e2m_8_2023_-80.53_-66.94_38.92_47.47northeastern_buffered_filtered_0.9.nc
│   └── Processed
│       ├── era5_2019_-83_-65_37_49.nc
│       ├── era5_2020_-83_-65_37_49.nc
│       ├── era5_2021_-83_-65_37_49.nc
│       ├── era5_2022_-83_-65_37_49.nc
│       └── era5_2023_-83_-65_37_49.nc
├── Shapefiles
│   └── Regions
│       ├── northeastern_buffered.cpg
│       ├── northeastern_buffered.dbf
│       ├── northeastern_buffered.prj
│       ├── northeastern_buffered.shp
│       └── northeastern_buffered.shx
└── madis
    ├── processed
    │   └── Meta--2019--2023
    │       ├── madis_2019_-80.53_-66.94_38.92_47.47northeastern_buffered_filtered_0.9.nc
    │       ├── madis_2020_-80.53_-66.94_38.92_47.47northeastern_buffered_filtered_0.9.nc
    │       ├── madis_2021_-80.53_-66.94_38.92_47.47northeastern_buffered_filtered_0.9.nc
    │       ├── madis_2022_-80.53_-66.94_38.92_47.47northeastern_buffered_filtered_0.9.nc
    │       └── madis_2023_-80.53_-66.94_38.92_47.47northeastern_buffered_filtered_0.9.nc
    └── stations
        ├── stations_2019_2023_-80.53_-66.94_38.92_47.47northeastern_buffered.cpg
        ├── stations_2019_2023_-80.53_-66.94_38.92_47.47northeastern_buffered.dbf
        ├── stations_2019_2023_-80.53_-66.94_38.92_47.47northeastern_buffered.prj
        ├── stations_2019_2023_-80.53_-66.94_38.92_47.47northeastern_buffered.shp
        ├── stations_2019_2023_-80.53_-66.94_38.92_47.47northeastern_buffered.shx
        ├── stations_2019_2023_-80.53_-66.94_38.92_47.47northeastern_buffered_filtered_0.9.cpg
        ├── stations_2019_2023_-80.53_-66.94_38.92_47.47northeastern_buffered_filtered_0.9.dbf
        ├── stations_2019_2023_-80.53_-66.94_38.92_47.47northeastern_buffered_filtered_0.9.prj
        ├── stations_2019_2023_-80.53_-66.94_38.92_47.47northeastern_buffered_filtered_0.9.shp
        └── stations_2019_2023_-80.53_-66.94_38.92_47.47northeastern_buffered_filtered_0.9.shx
```
