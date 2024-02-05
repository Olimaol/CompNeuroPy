from CompNeuroPy import run_script_parallel, load_variables, print_df
from itertools import product
import pandas as pd

MAX_EVALS = 500
N_PER_COND = 50
PARALLEL_WORKERS = 4


def FILE_APPENDIX(method, features, sim_id):
    return f"_{method}_{features}_{sim_id}"


if __name__ == "__main__":
    ### define the arguments for the script
    method_list = ["hyperopt", "deap", "bads"]
    features_list = ["voltage", "efel"]
    args_list = [
        [str(sim_id), method, features]
        for sim_id, method, features in product(
            range(N_PER_COND), method_list, features_list
        )
    ]
    ### run the script in parallel
    run_script_parallel(
        "benchmark_izhikevich.py",
        PARALLEL_WORKERS,
        args_list,
    )

    # ### load the benchmark variables TODO
    # benchmark_dict = {
    #     "method": [],
    #     "features": [],
    #     "best_loss": [],
    #     "best_loss_time": [],
    #     "best_loss_evals": [],
    #     "parameter_difference": [],
    # }
    # for sim_id, method, features in args_list:
    #     file_appendix = FILE_APPENDIX(method, features, sim_id)
    #     variables_dict = load_variables(
    #         name_list=[f"benchmark_variables{file_appendix}"],
    #         path="benchmark_izhikevich_results/",
    #     )[f"benchmark_variables{file_appendix}"]
    #     benchmark_dict["method"].append(method)
    #     benchmark_dict["features"].append(features)
    #     benchmark_dict["best_loss"].append(variables_dict["best_loss"])
    #     benchmark_dict["best_loss_time"].append(variables_dict["best_loss_time"])
    #     benchmark_dict["best_loss_evals"].append(variables_dict["best_loss_evals"])
    #     benchmark_dict["parameter_difference"].append(
    #         variables_dict["parameter_difference"]
    #     )
    # df = pd.DataFrame(benchmark_dict)
    # print_df(df)
    # df = df.sort_values(by=["method", "features"])
    # print_df(df)
