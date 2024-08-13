from test2 import OPTIMIZE_FOLDER, difference_binomial_normal
from CompNeuroPy import DeapCma, load_variables, save_variables
import numpy as np
import sys


def evaluate_function(population):
    loss_list = []
    ### the population is a list of individuals which are lists of parameters
    for individual in population:
        loss_of_individual = difference_binomial_normal(
            mean_shift=individual[0], std_scale=individual[1], n=n, p=p
        )
        loss_list.append((loss_of_individual,))
    return loss_list


if __name__ == "__main__":
    # Load the variables
    variables = load_variables(
        name_list=[
            "n",
            "p",
        ],
        path=OPTIMIZE_FOLDER,
    )
    n = variables["n"]
    p = variables["p"]

    ### bounds for optimized parameters
    shift_mean_bounds = [-1, 1]
    scale_std_bounds = [0.5, 2]
    lb = np.array([shift_mean_bounds[0], scale_std_bounds[0]])
    ub = np.array([shift_mean_bounds[1], scale_std_bounds[1]])
    p0 = np.random.default_rng().uniform(
        low=lb + 0.25 * (ub - lb), high=ub - 0.25 * (ub - lb), size=2
    )

    ### create an instance of the DeapCma class
    deap_cma = DeapCma(
        lower=lb,
        upper=ub,
        evaluate_function=evaluate_function,
        param_names=["mean_shift", "std_scale"],
        hard_bounds=True,
        display_progress_bar=False,
    )

    ### run the optimization
    deap_cma_result = deap_cma.run(max_evals=1000)

    ### get the optimized parameters and best error
    mean_shift_opt = deap_cma_result["mean_shift"]
    std_scale_opt = deap_cma_result["std_scale"]
    error_opt = deap_cma_result["best_fitness"]

    # Save the variables
    save_variables(
        name_list=[
            f"mean_shift_opt_{sys.argv[1]}",
            f"std_scale_opt_{sys.argv[1]}",
            f"error_opt_{sys.argv[1]}",
        ],
        variable_list=[mean_shift_opt, std_scale_opt, error_opt],
        path=OPTIMIZE_FOLDER,
    )
