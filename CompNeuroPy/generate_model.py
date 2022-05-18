from CompNeuroPy.neuron_models import H_and_H_Bischop, H_and_H_Corbit2016
from CompNeuroPy.model_functions import compile_in_folder
from ANNarchy import Population

def H_and_H_model_Bischop(do_compile=False, compile_folder_name='annarchy_HH_Bischop'):
    """
        generates one neuron of the H & H neuron model of Bischop et al. (2012) and optionally compiles the network
        
        returns a list of the names of the populations (for later access)
    """
    pop   = Population(1, neuron=H_and_H_Bischop, name='HH_Bischop')

    if do_compile:
        compile_in_folder(compile_folder_name)
    
    return {'populations':['HH_Bischop'], 'projections':[]}
    
    

def H_and_H_model_Corbit2016(do_compile=False, compile_folder_name='annarchy_HH_Corbit2016'):
    """
        generates one neuron of the H & H neuron model of Corbit et al. (2016) and optionally compiles the network
        
        returns a list of the names of the populations (for later access)
    """
    pop   = Population(1, neuron=H_and_H_Corbit2016, name='HH_Corbit2016')

    if do_compile:
        compile_in_folder(compile_folder_name)
    
    return {'populations':['HH_Corbit2016'], 'projections':[]}
