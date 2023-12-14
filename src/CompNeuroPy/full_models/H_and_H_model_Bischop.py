from ..generate_model import CompNeuroModel
from CompNeuroPy.neuron_models import HHneuronBischop
from ANNarchy import Population


class HHmodelBischop(CompNeuroModel):
    """
    Generates one neuron of the Hodgkin & Huxley neuron model of
    [Bischop et al. (2012)](https://doi.org/10.3389/fnmol.2012.00078) and optionally
    creates/compiles the network.
    """

    def __init__(
        self,
        name="single_HH_Bischop",
        do_create=True,
        do_compile=True,
        compile_folder_name="annarchy_single_HH_Bischop",
    ):
        """
        Args:
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
        Population(1, neuron=HHneuronBischop, name="HH_Bischop")


### old object for backwards compatibility
H_and_H_model_Bischop = HHmodelBischop
