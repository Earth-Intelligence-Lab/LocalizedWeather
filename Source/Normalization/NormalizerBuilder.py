# Author: Qidong Yang & Jonathan Giezendanner

import numpy as np

from Normalization.Normalizers import MinMaxNormalizer, ABNormalizer, StandardNormalizer
from Settings.Settings import ScalingType, EnvVariables


def get_normalizers(Data_List, external_object, madis_vars, external_vars, scaling=ScalingType.Standard):
    ##### Get Normalizer #####
    madis_mins = dict()
    madis_maxs = dict()
    madis_means = dict()
    madis_stds = dict()
    madis_ns = dict()
    madis_norm_dict = dict()

    for madis_var in madis_vars:
        madis_var_mins = []
        madis_var_maxs = []
        madis_var_means = []
        madis_var_stds = []
        madis_var_ns = []

        for Data in Data_List:
            madis_var_mins.append(Data.madis_mins_dict[madis_var])
            madis_var_maxs.append(Data.madis_maxs_dict[madis_var])
            madis_var_means.append(Data.madis_maxs_dict[madis_var])
            madis_var_stds.append(Data.madis_maxs_dict[madis_var])
            madis_var_ns.append(Data.madis_maxs_dict[madis_var])

        madis_var_min = np.min(np.array(madis_var_mins))
        madis_var_max = np.max(np.array(madis_var_maxs))
        madis_var_n = np.sum(np.array(madis_var_ns))
        madis_var_mean = np.sum(np.array(madis_var_means) * np.array(madis_var_ns)) / madis_var_n

        madis_var_std = np.sqrt(
            np.sum(
                np.array(madis_var_ns) * (np.array(madis_var_stds) ** 2)
                + np.array(madis_var_ns) * (np.array(madis_var_means) ** 2)
            )
            / madis_var_n - madis_var_mean ** 2
        )

        madis_mins[madis_var] = madis_var_min
        madis_maxs[madis_var] = madis_var_max
        madis_means[madis_var] = madis_var_mean
        madis_stds[madis_var] = madis_var_std
        madis_ns[madis_var] = madis_var_n

    if (EnvVariables.u in madis_vars) and (EnvVariables.v in madis_vars):
        madis_u_min = madis_mins[EnvVariables.u]
        madis_u_max = madis_maxs[EnvVariables.u]

        madis_v_min = madis_mins[EnvVariables.v]
        madis_v_max = madis_maxs[EnvVariables.v]

        madis_scale = np.max(np.array([np.abs(madis_u_min), np.abs(madis_v_min), madis_u_max, madis_v_max]))
        madis_std = np.max(np.array([madis_stds[EnvVariables.u], madis_stds[EnvVariables.v]]))

        if scaling == ScalingType.MinMax:
            madis_norm_dict[EnvVariables.u] = ABNormalizer(-madis_scale, madis_scale, -1.0, 1.0)
            madis_norm_dict[EnvVariables.v] = ABNormalizer(-madis_scale, madis_scale, -1.0, 1.0)
        elif scaling == ScalingType.Standard:
            madis_norm_dict[EnvVariables.u] = StandardNormalizer(0.0, madis_std)
            madis_norm_dict[EnvVariables.v] = StandardNormalizer(0.0, madis_std)

    for madis_var in madis_vars:
        if (madis_var == EnvVariables.u) or (madis_var == EnvVariables.v):
            continue

        if scaling == ScalingType.MinMax:
            madis_norm_dict[madis_var] = MinMaxNormalizer(madis_mins[madis_var], madis_maxs[madis_var])
        elif scaling == ScalingType.Standard:
            madis_norm_dict[madis_var] = StandardNormalizer(madis_means[madis_var], madis_stds[madis_var])

    if external_object is not None:

        external_mins = dict()
        external_maxs = dict()
        external_means = dict()
        external_stds = dict()
        external_ns = dict()
        external_norm_dict = dict()

        for external_var in external_vars:
            external_var_mins = []
            external_var_maxs = []
            external_var_means = []
            external_var_stds = []
            external_var_ns = []

            for Data in Data_List:
                external_var_mins.append(Data.external_mins_dict[external_var])
                external_var_maxs.append(Data.external_maxs_dict[external_var])
                external_var_means.append(Data.external_maxs_dict[external_var])
                external_var_stds.append(Data.external_maxs_dict[external_var])
                external_var_ns.append(Data.external_maxs_dict[external_var])

            external_var_min = np.min(np.array(external_var_mins))
            external_var_max = np.max(np.array(external_var_maxs))
            external_var_n = np.sum(np.array(external_var_ns))
            external_var_mean = np.sum(np.array(external_var_means) * np.array(external_var_ns)) / external_var_n

            external_var_std = np.sqrt(
                np.sum(
                    np.array(external_var_ns) * (np.array(external_var_stds) ** 2)
                    + np.array(external_var_ns) * (np.array(external_var_means) ** 2)
                )
                / external_var_n - external_var_mean ** 2
            )

            external_mins[external_var] = external_var_min
            external_maxs[external_var] = external_var_max
            external_means[external_var] = external_var_mean
            external_stds[external_var] = external_var_std
            external_ns[external_var] = external_var_n

        if (EnvVariables.u in external_vars) and (EnvVariables.v in external_vars):
            external_u_min = np.min(np.array(external_mins[EnvVariables.u]))
            external_u_max = np.max(np.array(external_maxs[EnvVariables.u]))

            external_v_min = np.min(np.array(external_mins[EnvVariables.v]))
            external_v_max = np.max(np.array(external_maxs[EnvVariables.v]))

            external_scale = np.max(
                np.array([np.abs(external_u_min), np.abs(external_v_min), external_u_max, external_v_max]))
            external_std = np.max(np.array([external_stds[EnvVariables.u], external_stds[EnvVariables.v]]))

            if scaling == ScalingType.MinMax:
                external_norm_dict[EnvVariables.u] = ABNormalizer(-external_scale, external_scale, -1.0, 1.0)
                external_norm_dict[EnvVariables.v] = ABNormalizer(-external_scale, external_scale, -1.0, 1.0)
            elif scaling == ScalingType.Standard:
                external_norm_dict[EnvVariables.u] = StandardNormalizer(0.0, external_std)
                external_norm_dict[EnvVariables.v] = StandardNormalizer(0.0, external_std)

        for external_var in external_vars:
            if (external_var == EnvVariables.u) or (external_var == EnvVariables.v):
                continue

            if scaling == ScalingType.MinMax:
                external_norm_dict[external_var] = MinMaxNormalizer(external_mins[external_var],
                                                                    external_maxs[external_var])
            elif scaling == ScalingType.Standard:
                external_norm_dict[external_var] = StandardNormalizer(external_means[external_var],
                                                                      external_stds[external_var])
    else:
        external_norm_dict = None

    return madis_norm_dict, external_norm_dict
