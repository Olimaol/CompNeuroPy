"""
This example demonstrates how to use the OptNeuron class to fit an ANNarchy neuron
model to the dynamics of another ANNarchy neuron model in a specific experiment.

The experiment and variable_bounds used are imported from the other example
'run_opt_neuron_from_data.py'.
"""
from CompNeuroPy import CompNeuroExp, rmse
from CompNeuroPy.opt_neuron import OptNeuron
from ANNarchy import Neuron

### import the experiment and variables_bounds
from run_opt_neuron_from_data import my_exp, variables_bounds
from run_opt_neuron_from_data import my_neuron as simple_neuron


### for this example we want to fit a simple neuron model to replicate the dynamics of a
### more complex neuron model, the simple model is imported from the other example
### 'run_opt_neuron_from_data.py' and the complex model is defined here
complex_neuron = Neuron(
    parameters="""
        I_app = 0
        m0 = 1
        m1 = 2
        m2 = 3
        n0 = 1
        n1 = 0
        n2 = -1
    """,
    equations="""
        r = m0*I_app + n0 + m1*I_app + n1 + m2*I_app + n2
    """,
)


### Next, the OptNeuron class needs a function to calculate the loss.
def get_loss(
    results_ist: CompNeuroExp._ResultsCl, results_soll: CompNeuroExp._ResultsCl
):
    """
    Function which has to have the arguments results_ist and results_soll and should
    calculate and return the loss. This structure is needed for the OptNeuron class.

    Args:
        results_ist (object):
            the results object returned by the run function of experiment (see above),
            conducting the experiment with the optimized neuron model
        results_soll (any):
            the results object returned by the run function of experiment (see above),
            conducting the experiment with the target neuron model

    Returns:
        loss (float or list of floats):
            the loss
    """

    ### get the recordings and other important things from the results_ist (results
    ### generated during the optimization using the defrined CompNeuroExp from above)
    rec_ist = results_ist.recordings
    pop_ist = results_ist.data["population_name"]
    rec_soll = results_soll.recordings
    pop_soll = results_soll.data["population_name"]

    ### the get_loss function should always calculate the loss for neuron rank 0! For
    ### both, the target and the optimized neuron model.
    neuron = 0

    ### get the data for calculating the loss from the recordings of the
    ### target neuron model
    v_soll_0 = rec_soll[0][pop_soll + ";r"][:, neuron]
    v_soll_1 = rec_soll[1][pop_soll + ";r"][:, neuron]

    ### get the data for calculating the loss from the recordings of the
    ### optimized neuron model
    v_ist_0 = rec_ist[0][pop_ist + ";r"][:, neuron]
    v_ist_1 = rec_ist[1][pop_ist + ";r"][:, neuron]

    ### calculate the loss, e.g. the root mean squared error
    rmse1 = rmse(v_soll_0, v_ist_0)
    rmse2 = rmse(v_soll_1, v_ist_1)

    ### return the loss, one can return a singel value or a list of values which will
    ### be summed during the optimization
    return [rmse1, rmse2]


def main():
    ### define optimization
    opt = OptNeuron(
        experiment=my_exp,
        get_loss_function=get_loss,
        variables_bounds=variables_bounds,
        neuron_model=simple_neuron,
        target_neuron_model=complex_neuron,
        time_step=1,
        compile_folder_name="annarchy_opt_neuron_example_from_neuron",
        method="deap",
        record=["r"],
    )

    ### run the optimization, define how often the experiment should be repeated
    fit = opt.run(max_evals=100, results_file_name="best_from_neuron")

    ### print optimized parameters, we should get around a=6 and b=0
    print("a", fit["a"])
    print("b", fit["b"])
    print(list(fit.keys()))


if __name__ == "__main__":
    main()
