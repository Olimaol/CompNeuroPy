from ..generate_model import generate_model
from CompNeuroPy.neuron_models import H_and_H_Corbit
from ANNarchy import Population

class H_and_H_model_Corbit(generate_model):
    """
        generates one neuron of the H & H neuron model of Corbit et al. (2016) and optionally compiles the network
    """

    def __init__(self, name='single_HH_Corbit', do_create=True, do_compile=True, compile_folder_name='annarchy_single_HH_Corbit'):
        """
            runs the standard init but with already predefined model_creation_function and description
            one can still adjust name, do_compile and compile_folder_name
        """
        description = 'One population "HH_Corbit" with a single neuron of the H & H neuron model of Corbit et al. (2016)'
        model_creation_function = self.__model_creation_function__
        super().__init__(model_creation_function=model_creation_function, name=name, description=description, do_create=do_create, do_compile=do_compile, compile_folder_name=compile_folder_name)
        
    def __model_creation_function__(self):
        pop = Population(1, neuron=H_and_H_Corbit, name='HH_Corbit')




