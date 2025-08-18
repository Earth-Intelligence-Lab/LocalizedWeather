from Settings.Settings import *


def PostProcessArgs(args):
    args.model_type = ModelType(args.model_type)
    args.network_construction_method = NetworkConstructionMethod(args.network_construction_method)
    args.loss_function_type = LossFunctionType(args.loss_function_type)
    args.interpolation_type = InterpolationType(args.interpolation_type)
    args.hrrr_analysis_only = args.hrrr_analysis_only == 1

    args.madis_vars_i = [EnvVariables(variable_k) for variable_k in args.madis_vars_i]
    args.madis_vars_o = [EnvVariables(variable_k) for variable_k in args.madis_vars_o]
    args.external_vars = [EnvVariables(variable_k) for variable_k in args.external_vars]

    return args
