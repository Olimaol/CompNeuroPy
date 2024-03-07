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


def _get_all_attributes(compartment_list: list[str]):
    """
    Get the attributes of specified populations and projections.

    Args:
        compartment_list (list[str]):
            List of compartment names.

    Returns:
        attributes (dict of dicts):
            Dictionary with keys "populations" and "projections" and values dicts of
            attributes of populations and projections, respectively.
    """
    attributes = {
        "populations": {},
        "projections": {},
    }
    ### check if compartments are in model
    for compartment in compartment_list:
        if (
            compartment
            not in get_full_model()["populations"] + get_full_model()["projections"]
        ):
            raise ValueError(f"Compartment {compartment} not found in model")
    ### get attributes of populations
    for pop in populations():
        if pop.name not in compartment_list:
            continue
        attributes["populations"][pop.name] = {
            param_name: getattr(pop, param_name) for param_name in pop.attributes
        }
    ### get attributes of projections
    for proj in projections():
        if proj.name not in compartment_list:
            continue
        attributes["projections"][proj.name] = {
            param_name: getattr(proj, param_name) for param_name in proj.attributes
        }
    return attributes


def _set_all_attributes(attributes: dict, parameters: bool):
    """
    Set the attributes of all populations and projections given in attributes dict.

    Args:
        attributes (dict of dicts):
            Dictionary with keys "populations" and "projections" and values dicts of
            attributes of populations and projections, respectively.
        parameters (bool):
            If True, set parameters and variables, else only set variables.
    """
    ### set attributes of populations
    for pop in populations():
        ### skip populations which are not in attributes
        if pop.name not in attributes["populations"].keys():
            continue
        for param_name, param_value in attributes["populations"][pop.name].items():
            ### skip parameters if parameters is False
            if param_name in pop.parameters and not parameters:
                continue
            setattr(pop, param_name, param_value)
    ### set attributes of projections
    for proj in projections():
        ### skip projections which are not in attributes
        if proj.name not in attributes["projections"].keys():
            continue
        for param_name, param_value in attributes["projections"][proj.name].items():
            ### skip parameters if parameters is False
            if param_name in proj.parameters and not parameters:
                continue
            setattr(proj, param_name, param_value)


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
