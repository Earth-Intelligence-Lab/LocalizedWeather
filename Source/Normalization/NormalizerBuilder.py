# Author: Qidong Yang & Jonathan Giezendanner

import numpy as np

from Normalization.Normalizers import MinMaxNormalizer, ABNormalizer


def get_normalizers(Data_List, era5_network):
    ##### Get Normalizer #####
    madis_u_mins = []
    madis_v_mins = []
    madis_temp_mins = []

    madis_u_maxs = []
    madis_v_maxs = []
    madis_temp_maxs = []

    for Data in Data_List:
        madis_u_mins.append(Data.madis_u_min)
        madis_v_mins.append(Data.madis_v_min)
        madis_temp_mins.append(Data.madis_temp_min)

        madis_u_maxs.append(Data.madis_u_max)
        madis_v_maxs.append(Data.madis_v_max)
        madis_temp_maxs.append(Data.madis_temp_max)

    madis_temp_min = np.min(np.array(madis_temp_mins))
    madis_temp_max = np.max(np.array(madis_temp_maxs))

    madis_temp_normalizer = MinMaxNormalizer(madis_temp_min, madis_temp_max)

    madis_u_min = np.min(np.array(madis_u_mins))
    madis_v_min = np.min(np.array(madis_v_mins))
    madis_u_max = np.max(np.array(madis_u_maxs))
    madis_v_max = np.max(np.array(madis_v_maxs))
    madis_scale = np.max(np.array([np.abs(madis_u_min), np.abs(madis_v_min), madis_u_max, madis_v_max]))

    madis_u_normalizer = ABNormalizer(-madis_scale, madis_scale, -1.0, 1.0)
    madis_v_normalizer = ABNormalizer(-madis_scale, madis_scale, -1.0, 1.0)
    madis_norm_dict = {'u': madis_u_normalizer, 'v': madis_v_normalizer, 'temp': madis_temp_normalizer}

    if era5_network is not None:
        era5_u_mins = []
        era5_v_mins = []
        era5_temp_mins = []

        era5_u_maxs = []
        era5_v_maxs = []
        era5_temp_maxs = []

        for Data in Data_List:
            era5_u_mins.append(Data.era5_u_min)
            era5_v_mins.append(Data.era5_v_min)
            era5_temp_mins.append(Data.era5_temp_min)

            era5_u_maxs.append(Data.era5_u_max)
            era5_v_maxs.append(Data.era5_v_max)
            era5_temp_maxs.append(Data.era5_temp_max)

        era5_u_min = np.min(np.array(era5_u_mins))
        era5_v_min = np.min(np.array(era5_v_mins))
        era5_temp_min = np.min(np.array(era5_temp_mins))
        era5_u_max = np.max(np.array(era5_u_maxs))
        era5_v_max = np.max(np.array(era5_v_maxs))
        era5_temp_max = np.max(np.array(era5_temp_maxs))
        era5_scale = np.max(np.array([np.abs(era5_u_min), np.abs(era5_v_min), era5_u_max, era5_v_max]))

        era5_u_normalizer = MinMaxNormalizer(-era5_scale, era5_scale)
        era5_v_normalizer = MinMaxNormalizer(-era5_scale, era5_scale)

        era5_temp_normalizer = MinMaxNormalizer(era5_temp_min, era5_temp_max)

        era5_norm_dict = {'u': era5_u_normalizer, 'v': era5_v_normalizer, 'temp': era5_temp_normalizer}
    else:
        era5_norm_dict = None

    return madis_norm_dict, era5_norm_dict
