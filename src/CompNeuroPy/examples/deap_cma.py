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


def get_source_solutions(lb, ub):
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
    ### create random solutions
    source_solutions_parameters = np.random.uniform(0, 1, (100, 3)) * (ub - lb) + lb
    ### evaluate the random solutions
    source_solutions_losses = evaluate_function(source_solutions_parameters)
    ### create a list of tuples, where each tuple contains the parameters of an
    ### individual and its loss
    source_solutions = [
        (source_solutions_parameters[idx], source_solutions_losses[idx][0])
        for idx in range(len(source_solutions_parameters))
    ]
    ### only use the best 10 as source solutions
    source_solutions = sorted(source_solutions, key=lambda x: x[1])[:10]

    return source_solutions


def main():
    ### define lower bounds of paramters to optimize
    lb = np.array([-10, -10, 0])

    ### define upper bounds of paramters to optimize
    ub = np.array([10, 15, 5])

    ### create an "minimal" instance of the DeapCma class
    deap_cma = DeapCma(
        lower=lb,
        upper=ub,
        evaluate_function=evaluate_function,
    )

    ### create an instance of the DeapCma class using all optional attributes
    ### to initialize one could give a p0 array (same shape as lower and upper) and a
    ### sig0 value or use source solutions (as shown here)
    deap_cma_optional = DeapCma(
        lower=lb,
        upper=ub,
        evaluate_function=evaluate_function,
        max_evals=1000,
        p0=None,
        sig0=None,
        param_names=["a", "b", "c"],
        learn_rate_factor=1,
        damping_factor=1,
        verbose=True,
        plot_file="logbook_optional.png",
        cma_params_dict={},
        source_solutions=get_source_solutions(lb=lb, ub=ub),
        hard_bounds=True,
    )

    ### run the optimization, since max_evals was not defined during initialization of
    ### the DeapCma instance, it has to be defined here
    ### it automatically saves a plot file showing the loss over the generations
    deap_cma_result = deap_cma.run(max_evals=1000)

    ### run the optimization with all optional attributes
    deap_cma_optional_result = deap_cma_optional.run(verbose=False)

    ### print the best parameters and its loss, since we did not define the names of the
    ### parameters during initialization of the DeapCma instance, the names are param0,
    ### param1, param2, also print everything that is in the dict returned by the run
    best_param_dict = {
        param_name: deap_cma_result[param_name]
        for param_name in ["param0", "param1", "param2"]
    }
    print("\nFirst (minimal) optimization:")
    print(f"Dict from run function contains: {list(deap_cma_result.keys())}")
    print(f"Best parameters: {best_param_dict}")
    print(f"Loss of best parameters: {deap_cma_result['best_fitness']}\n")

    ### print the same for the second optimization
    best_param_dict = {
        param_name: deap_cma_optional_result[param_name]
        for param_name in ["a", "b", "c"]
    }
    print("Second optimization (with all optional attributes):")
    print(f"Dict from run function contains: {list(deap_cma_optional_result.keys())}")
    print(f"Best parameters: {best_param_dict}")
    print(f"Loss of best parameters: {deap_cma_optional_result['best_fitness']}")

    return 1


if __name__ == "__main__":
    main()
