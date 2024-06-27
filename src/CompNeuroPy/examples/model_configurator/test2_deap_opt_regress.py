from CompNeuroPy import DeapCma, load_variables, save_variables
import numpy as np
from test2 import deap_opt_regress_path, curve_fit_func
import sys

# Load the variables
variables = load_variables(
    name_list=[
        "x",
        "y",
        "z",
    ],
    path=deap_opt_regress_path,
)
x = variables["x"]
y = variables["y"]
z = variables["z"]


param_names = [
    "g0",
    "g1",
    "g2",
    "g3",
    "g4",
    "g5",
    "g6",
    "g7",
    "g8",
    "g9",
    "g10",
    "g11",
    "g12",
    "g13",
    "g14",
    "p0",
    # "p1",
    # "p2",
    # "p3",
    # "p4",
    # "p5",
    # "p6",
    # "p7",
    # "p8",
    # "p9",
    # "p10",
    # "p11",
    # "p12",
    # "p13",
    # "p14",
    # "p15",
    # "p16",
    # "p17",
    # "p18",
    # "p19",
    # "p20",
]


def curve_fit_evaluate_function(population):
    loss_list = []
    ### the population is a list of individuals which are lists of parameters
    for individual in population:
        loss_of_individual = curve_fit_objective_function(individual)
        loss_list.append((loss_of_individual,))
    return loss_list


def curve_fit_objective_function(individual):
    is_data = curve_fit_func((x, y), *individual)
    target_data = z
    return np.sum((is_data - target_data) ** 2)


deap_cma = DeapCma(
    lower=np.array([-1] * len(param_names)),
    upper=np.array([1] * len(param_names)),
    evaluate_function=curve_fit_evaluate_function,
    param_names=param_names,
    hard_bounds=False,
    display_progress_bar=False,
)
deap_cma_result = deap_cma.run(max_evals=2000)
popt = [deap_cma_result[param_name] for param_name in param_names]

# Save the variables
save_variables(
    name_list=[f"popt_{sys.argv[1]}", f"best_fitness_{sys.argv[1]}"],
    variable_list=[popt, deap_cma_result["best_fitness"]],
    path=deap_opt_regress_path,
)
