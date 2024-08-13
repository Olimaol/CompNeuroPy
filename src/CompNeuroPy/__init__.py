### ANNarchy
import os
import sys
from contextlib import contextmanager


@contextmanager
def suppress_stdout():
    """
    Suppresses the print output of a function

    Example:
        ```python
        with suppress_stdout():
            print("this will not be printed")
        ```
    """
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


with suppress_stdout():
    import ANNarchy as ann
    from ANNarchy.core import ConnectorMethods as ann_ConnectorMethods

    if ann.__version__ >= "4.8":
        from ANNarchy.intern.NetworkManager import NetworkManager as ann_NetworkManager
    else:
        from ANNarchy.core import Global as ann_Global

    from ANNarchy.core import Random as ann_Random

### functions
from CompNeuroPy.analysis_functions import (
    my_raster_plot,
    get_nanmean,
    get_nanstd,
    get_population_power_spectrum,
    get_power_spektrum_from_time_array,
    get_pop_rate,
    get_number_of_zero_decimals,
    get_number_of_decimals,
    sample_data_with_timestep,
    rmse,
    rsse,
    get_minimum,
    get_maximum,
    PlotRecordings,
)
from CompNeuroPy.extra_functions import (
    print_df,
    flatten_list,
    remove_key,
    suppress_stdout,
    sci,
    Cmap,
    create_cm,
    DecisionTree,
    DecisionTreeNode,
    evaluate_expression_with_dict,
    VClampParamSearch,
    DeapCma,
    InteractivePlot,
    efel_loss,
    RNG,
    find_x_bound,
)
from CompNeuroPy.model_functions import (
    compile_in_folder,
    annarchy_compiled,
    get_full_model,
    cnp_clear,
)
from CompNeuroPy.simulation_functions import (
    current_stim,
    current_step,
    current_ramp,
    increasing_current,
    attr_sim,
    attribute_step,
    attr_ramp,
    increasing_attr,
    SimulationEvents,
)
from CompNeuroPy.system_functions import (
    clear_dir,
    create_dir,
    save_variables,
    load_variables,
    timing_decorator,
    run_script_parallel,
    create_data_raw_folder,
    Logger,
)
from CompNeuroPy.simulation_requirements import req_pop_attr, ReqPopHasAttr
from CompNeuroPy.statistic_functions import anova_between_groups

### classes
from CompNeuroPy.monitors import CompNeuroMonitors
from CompNeuroPy.experiment import CompNeuroExp
from CompNeuroPy.generate_model import CompNeuroModel
from CompNeuroPy.generate_simulation import CompNeuroSim
from CompNeuroPy.dbs import DBSstimulator

### modules
### for opt_neuron you need to install torch, sbi, and hyperopt
# from CompNeuroPy.opt_neuron as OptNeuron
