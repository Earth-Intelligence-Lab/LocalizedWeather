# Localized Weather GNN

This repository contains the code and data for the paper "Multi-Modal Graph Neural Networks for Localized Off-Grid Weather Forecasting".

The paper presents a novel multi-modal graph neural network (GNN) that downscales gridded weather forecasts, such as ERA5, to provide accurate off-grid predictions. The model leverages both ERA5 data and local weather station observations from MADIS to make predictions that reflect both large-scale atmospheric dynamics and local weather patterns.

The model is evaluated on a surface wind prediction task and shows significant improvement over baseline methods, including ERA5 interpolation and a multi-layer perceptron.

Use the following citation when these data or model are used:
> Yang, Q.; Giezendanner, J.; Civitarese, D. S.; Jakubik, J.; 
,Schmitt E.; Chandra, A.; Vila, J.; Hohl, D.; Hill, C.; Watson, C.; Wang, S.; Multi-modal graph neural networks for localized off-grid weather forecasting. arXiv, October 2024. https://arxiv.org/



# Model and data
## Data

The data for training and inference can be found at [doi://](https://doi).

The following data is available:
- Shapefile of the Northeastern United States (NE-US, extracted from [NWS](https://www.weather.gov/gis/USStates))
- Shapefile containing the location and number of observations (2019-2023) of the MADIS stations in NE-US
- Processed hourly averaged [MADIS](https://madis.ncep.noaa.gov/) data for the NE-US (2019-2023)
- [ERA5](https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation) data for the NE-US (2019-2023), gridded and interpolated

For MADIS and ERA5, the following variables are available:
- u and v component of wind vector at 10 meters above ground
- temperature at 2 meters above ground
- dewpoint at 2 meters above ground
- solar radiation

## Code
The code is organised as follows (in `Source/`):
- `GNN_Launcher.py`, `GNN_NotebookLauncher.ipynb` and `GNN_SLURM_job_launcher.py` are three different launchers all eventually pointing at `GNN_Main.py` (the arguments from the slurm job launcher are parsed by `GNN_arg_parser.py`)
- `GNN_Main.py` contains the main code loop
- `EvaluateModel` contains the code for the evaluation of the model, as well as the back propagation
- the folder `Dataloader/` contains the data loaders for MADIS and ERA5, and the combination of both, and the folder `Network/` the code for the network construction, for both the internal (MADIS) and external (ERA5 to MADIS) connections
- `Modules/GNN/MPNN.py` contains the code for the heterogeneous message passing neural network and calls `GNN_Layer_Internal/External.py`, the message passing sequences between the networks



