"""
This example demonstrates how to use the DeapCma class to optimize parameters.
"""

from CompNeuroPy import DeapCma
import numpy as np


### for DeapCma we need to define the evaluate_function
def evaluate_function(population):
    """
    Calculate the loss for a population of individuals.

    Args:
        population (np.ndarray):
            population of individuals (i.e., parameter sets) to evaluate

    Returns:
        loss_values (list[tuple]):
            list of tuples, where each tuple contains the loss for an individual of the
            population
    """
    loss_list = []
    ### the population is a list of individuals
    for individual in population:
        ### the individual is a list of parameters
        p0, p1, p2 = individual
        ### calculate the loss of the individual
        loss_of_individual = float((p0 - 3) ** 2 + (p1 - 7) ** 2 + (p2 - (-2)) ** 2)
        ### insert the loss of the individual into the list of tuples
        loss_list.append((loss_of_individual,))

    return loss_list


def get_source_solutions():
    """
    DeapCma can use source solutions to initialize the optimization process. This
    function returns an example of source solutions.

    Source solutions are a list of tuples, where each tuple contains the parameters of
    an individual (np.ndarray) and its loss (float).

    Returns:
        source_solutions (list[tuple]):
            list of tuples, where each tuple contains the parameters of an individual
            and its loss
    """
    source_solutions_parameters = np.array(
        [
            [1, 2, 3],
            [3, 5, 3],
            [5, 7, 3],
            [7, 9, 3],
            [9, 10, 3],
            [-1, -2, -3],
            [-3, -5, -3],
            [-5, -7, -3],
            [-7, -9, -3],
            [-9, -10, -3],
        ]
    )
    source_solutions_losses = evaluate_function(source_solutions_parameters)
    source_solutions = [
        (source_solutions_parameters[idx], source_solutions_losses[idx][0])
        for idx in range(len(source_solutions_parameters))
    ]

    return source_solutions


def main():
    ### define lower bounds of paramters to optimize
    lb = np.array([-10, -10, -10])

    ### define upper bounds of paramters to optimize
    ub = np.array([10, 10, 10])

    ### create an "minimal" instance of the DeapCma class
    deap_cma = DeapCma(
        lower=lb,
        upper=ub,
        evaluate_function=evaluate_function,
    )

    ### create an instance of the DeapCma class using all optional attributes
    ### to initialize one could give a p0 array (same shape as lower and upper) or use
    ### source solutions (as shown here)
    deap_cma_optional = DeapCma(
        lower=lb,
        upper=ub,
        evaluate_function=evaluate_function,
        max_evals=1000,
        p0=None,
        param_names=["a", "b", "c"],
        learn_rate_factor=0.5,
        damping_factor=0.5,
        verbose=False,
        plot_file="logbook_optional.png",
        cma_params_dict={},
        source_solutions=get_source_solutions(),
    )

    ### run the optimization, since max_evals was not defined during initialization of
    ### the DeapCma instance, it has to be defined here
    ### it automatically saves a plot file showing the loss over the generations
    deap_cma_result = deap_cma.run(max_evals=1000)

    ### run the optimization with all optional attributes
    deap_cma_optional_result = deap_cma_optional.run()

    ### print what deap_cma_result contains
    print(f"Dict from run function contains: {list(deap_cma_result.keys())}")

    ### print the best parameters and its loss, since we did not define the names of the
    ### parameters during initialization of the DeapCma instance, the names are param0,
    ### param1, param2
    best_param_dict = {
        param_name: deap_cma_result[param_name]
        for param_name in ["param0", "param1", "param2"]
    }
    print(f"Best parameters from first optimization: {best_param_dict}")
    print(
        f"Loss of best parameters from first optimization: {deap_cma_result['best_fitness']}"
    )

    ### print the same for the second optimization
    best_param_dict = {
        param_name: deap_cma_optional_result[param_name]
        for param_name in ["a", "b", "c"]
    }
    print(f"Best parameters from second optimization: {best_param_dict}")
    print(
        f"Loss of best parameters from second optimization: {deap_cma_optional_result['best_fitness']}"
    )

    return 1


if __name__ == "__main__":
    main()
