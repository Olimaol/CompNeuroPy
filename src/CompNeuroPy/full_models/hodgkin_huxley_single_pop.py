from ..generate_model import CompNeuroModel
from CompNeuroPy.neuron_models import (
    HHneuronCorbit,
    HHneuronCorbitSyn,
    HHneuronBischop,
    HHneuronBischopSyn,
)
from ANNarchy import Population


class HHmodelBischop(CompNeuroModel):
    """
    Generates a single population of the Hodgkin & Huxley neuron model of
    [Bischop et al. (2012)](https://doi.org/10.3389/fnmol.2012.00078) and optionally
    creates/compiles the network.
    """

    def __init__(
        self,
        pop_size=1,
        conductance_based_synapses=False,
        name="single_HH_Bischop",
        do_create=True,
        do_compile=True,
        compile_folder_name="annarchy_single_HH_Bischop",
    ):
        """
        Args:
            pop_size (int, optional):
                Number of neurons in the population. Default: 1.
            conductance_based_synapses (bool, optional):
                Whether the equations contain conductance based synapses for AMPA and
                GABA. Default: False.
            name (str, optional):
                Name of the model. Default: "single_HH_Bischop".
            do_create (bool, optional):
                Whether to create the model. Default: True.
            do_compile (bool, optional):
                Whether to compile the model. Default: True.
            compile_folder_name (str, optional):
                Name of the folder for the compiled model.
                Default: "annarchy_single_HH_Bischop".
        """
        ### set attributes
        self.pop_size = pop_size
        self.conductance_based_synapses = conductance_based_synapses
        # define description
        description = """
            One population "HH_Bischop" with a single neuron of the Hodgkin
            & Huxley neuron model of Bischop et al. (2012).
        """
        # initialize CompNeuroModel
        super().__init__(
            model_creation_function=self._bischop_2012_creation_function,
            name=name,
            description=description,
            do_create=do_create,
            do_compile=do_compile,
            compile_folder_name=compile_folder_name,
        )

    def _bischop_2012_creation_function(self):
        if self.conductance_based_synapses:
            Population(self.pop_size, neuron=HHneuronBischopSyn, name="HH_Bischop_syn")
        else:
            Population(self.pop_size, neuron=HHneuronBischop, name="HH_Bischop")


class HHmodelCorbit(CompNeuroModel):
    """
    Generates a single population of the Hodgkin & Huxley neuron model of
    [Corbit et al. (2016)](https://doi.org/10.1523/JNEUROSCI.0339-16.2016) and
    optionally creates/compiles the network.
    """

    def __init__(
        self,
        pop_size=1,
        conductance_based_synapses=False,
        name="single_HH_Corbit",
        do_create=True,
        do_compile=True,
        compile_folder_name="annarchy_single_HH_Corbit",
    ):
        """
        Args:
            pop_size (int, optional):
                Number of neurons in the population. Default: 1.
            conductance_based_synapses (bool, optional):
                Whether the equations contain conductance based synapses for AMPA and
                GABA. Default: False.
            name (str, optional):
                Name of the model. Default: "single_HH_Corbit".
            do_create (bool, optional):
                Whether to create the model. Default: True.
            do_compile (bool, optional):
                Whether to compile the model. Default: True.
            compile_folder_name (str, optional):
                Name of the folder for the compiled model.
                Default: "annarchy_single_HH_Corbit".
        """
        ### set attributes
        self.pop_size = pop_size
        self.conductance_based_synapses = conductance_based_synapses
        # define description
        description = """
            One population "HH_Bischop" with a single neuron of the Hodgkin
            & Huxley neuron model of Bischop et al. (2012).
        """
        # initialize CompNeuroModel
        super().__init__(
            model_creation_function=self._model_creation_function,
            name=name,
            description=description,
            do_create=do_create,
            do_compile=do_compile,
            compile_folder_name=compile_folder_name,
        )

    def _model_creation_function(self):
        if self.conductance_based_synapses:
            Population(self.pop_size, neuron=HHneuronCorbitSyn, name="HH_Corbit_syn")
        else:
            Population(self.pop_size, neuron=HHneuronCorbit, name="HH_Corbit")
