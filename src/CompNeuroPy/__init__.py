"""import CompNeuroPy.analysis_functions
import CompNeuroPy.extra_functions
import CompNeuroPy.generate_model
import CompNeuroPy.model_functions
import CompNeuroPy.neuron_models
import CompNeuroPy.opt_Izh_07
import CompNeuroPy.simulation_functions
import CompNeuroPy.system_functions
import CompNeuroPy.Monitors
import CompNeuroPy.Experiment"""
### functions
from CompNeuroPy.analysis_functions import *
from CompNeuroPy.extra_functions import *
from CompNeuroPy.model_functions import *
from CompNeuroPy.synapse_models import *
from CompNeuroPy.simulation_functions import *
from CompNeuroPy.system_functions import *
from CompNeuroPy.simulation_requirements import *

### classes
from CompNeuroPy.Monitors import Monitors
from CompNeuroPy.Experiment import Experiment
from CompNeuroPy.generate_model import generate_model
from CompNeuroPy.generate_simulation import generate_simulation
from CompNeuroPy.DBS import DBS_stimulator

### modules
### for opt_neuron you need to install torch, sbi, and hyperopt
import CompNeuroPy.opt_neuron as opt_neuron
