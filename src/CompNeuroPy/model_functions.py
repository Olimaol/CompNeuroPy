from ANNarchy import (
    compile,
    populations,
    projections,
    clear,
)
import os
from CompNeuroPy import system_functions as sf
from CompNeuroPy.generate_model import CompNeuroModel
from ANNarchy.core import Global


def compile_in_folder(folder_name, net=None, clean=False, silent=False):
    """
    Creates the compilation folder in annarchy_folders/ or uses existing ones. Compiles
    the current network.

    Args:
        folder_name (str):
            Name of the folder within annarchy_folders/
        net (ANNarchy network, optional):
            ANNarchy network. Default: None.
        clean (bool, optional):
            If True, the library is recompiled entirely, else only the changes since
            last compilation are compiled. Default: False.
        silent (bool, optional):
            Suppress output. Defaults to False.
    """
    sf.create_dir("annarchy_folders/" + folder_name, print_info=False)
    if isinstance(net, type(None)):
        compile("annarchy_folders/" + folder_name, clean=clean, silent=silent)
    else:
        net.compile("annarchy_folders/" + folder_name, clean=clean, silent=silent)
    if os.getcwd().split("/")[-1] == "annarchy_folders":
        os.chdir("../")


def annarchy_compiled(net_id=0):
    """
    Check if ANNarchy network was compiled.

    Args:
        net_id (int, optional):
            Network ID. Default: 0.
    """
    return Global._network[net_id]["compiled"]


def get_full_model():
    """
    Return all current population and projection names.

    Returns:
        model_dict (dict):
            Dictionary with keys "populations" and "projections" and values lists of
            population and projection names, respectively.
    """
    return {
        "populations": [pop.name for pop in populations()],
        "projections": [proj.name for proj in projections()],
    }


def cnp_clear(functions=True, neurons=True, synapses=True, constants=True):
    """
    Like clear with ANNarchy, but CompNeuroModel objects are also cleared.

    Args:
        functions (bool, optional):
            If True, all functions are cleared. Default: True.
        neurons (bool, optional):
            If True, all neurons are cleared. Default: True.
        synapses (bool, optional):
            If True, all synapses are cleared. Default: True.
        constants (bool, optional):
            If True, all constants are cleared. Default: True.
    """
    clear(functions=functions, neurons=neurons, synapses=synapses, constants=constants)
    for model_name in CompNeuroModel._initialized_models.keys():
        CompNeuroModel._initialized_models[model_name] = False
    for model_name in CompNeuroModel._compiled_models.keys():
        CompNeuroModel._compiled_models[model_name] = False


def _get_all_parameters():
    """
    Get the parameters of all populations and projections.

    Returns:
        parameters (dict of dicts):
            Dictionary with keys "populations" and "projections" and values dicts of
            parameters of populations and projections, respectively.
    """
    parameters = {
        "populations": {},
        "projections": {},
    }
    for pop in populations():
        parameters["populations"][pop.name] = {
            param_name: getattr(pop, param_name) for param_name in pop.parameters
        }
    for proj in projections():
        parameters["projections"][proj.name] = {
            param_name: getattr(proj, param_name) for param_name in proj.parameters
        }
    return parameters


def _set_all_parameters(parameters):
    """
    Set the parameters of all populations and projections.

    Args:
        parameters (dict of dicts):
            Dictionary with keys "populations" and "projections" and values dicts of
            parameters of populations and projections, respectively.
    """
    for pop in populations():
        for param_name, param_value in parameters["populations"][pop.name].items():
            setattr(pop, param_name, param_value)
    for proj in projections():
        for param_name, param_value in parameters["projections"][proj.name].items():
            setattr(proj, param_name, param_value)
