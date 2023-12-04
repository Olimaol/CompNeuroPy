from ..generate_model import generate_model
from CompNeuroPy.neuron_models import H_and_H_Corbit_syn
from ANNarchy import Population

class H_and_H_model_Corbit_syn(generate_model):
    """
        generates two neurons of the H & H neuron model of Corbit et al. (2016) with conductance based synapses and optionally compiles the network
    """

    def __init__(self, name='single_HH_Corbit_syn', do_create=True, do_compile=True, compile_folder_name='annarchy_single_HH_Corbit_syn'):
        """
            runs the standard init but with already predefined model_creation_function and description
            one can still adjust name, do_compile and compile_folder_name
        """
        description = 'One population "HH_Corbit_syn" with two neurons of the H & H neuron model of Corbit et al. (2016) with conductance based synapses'
        model_creation_function = self.__model_creation_function__
        super().__init__(model_creation_function=model_creation_function, name=name, description=description, do_create=do_create, do_compile=do_compile, compile_folder_name=compile_folder_name)
        
    def __model_creation_function__(self):
        pop = Population(2, neuron=H_and_H_Corbit_syn, name='HH_Corbit_syn')


