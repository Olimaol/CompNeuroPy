from ..generate_model import generate_model
from CompNeuroPy.neuron_models import H_and_H_Bischop
from ANNarchy import Population

class H_and_H_model_Bischop(generate_model):
    """
        generates one neuron of the H & H neuron model of Bischop et al. (2012) and optionally compiles the network
    """

    def __init__(self, name='single_HH_Bischop', do_create=True, do_compile=True, compile_folder_name='annarchy_single_HH_Bischop'):
        """
            runs the standard init but with already predefined model_creation_function and description
            one can still adjust name, do_create, do_compile and compile_folder_name
        """
        description = 'One population "HH_Bischop" with a single neuron of the H & H neuron model of Bischop et al. (2012)'
        model_creation_function = self.__model_creation_function__
        super().__init__(model_creation_function=model_creation_function, name=name, description=description, do_create=do_create, do_compile=do_compile, compile_folder_name=compile_folder_name)
        
    def __model_creation_function__(self):
        pop = Population(1, neuron=H_and_H_Bischop, name='HH_Bischop')
