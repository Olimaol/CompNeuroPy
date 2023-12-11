from ANNarchy import (
    compile,
    populations,
    projections,
    clear,
)
import os
from CompNeuroPy import system_functions as sf
from CompNeuroPy.generate_model import CompNeuroModel


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


def cnp_clear():
    """
    Like clear with ANNarchy, but CompNeuroModel objects are also cleared.
    """
    clear()
    for model_name in CompNeuroModel.initialized_models.keys():
        CompNeuroModel.initialized_models[model_name] = False
    for model_name in CompNeuroModel.compiled_models.keys():
        CompNeuroModel.compiled_models[model_name] = False
