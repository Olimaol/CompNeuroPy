import os
import traceback
import shutil
from time import time
import pickle
from functools import wraps
from joblib import Parallel, delayed
import inspect
import subprocess


def clear_dir(path):
    """
    Deletes all files and subdirectories in the specified folder.

    Args:
        path (str):
            Path to the folder to clear.
    """
    try:
        if not os.path.exists(path):
            print(f"The folder '{path}' does not exist.")
            return

        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception:
                print(traceback.format_exc())
                print(f"Failed to delete {file_path}")
    except Exception:
        print(traceback.format_exc())
        print(f"Failed to clear {path}")


def create_dir(path, print_info=False, clear=False):
    """
    Creates a directory.

    Args:
        path (str):
            Path to the directory to create.

        print_info (bool, optional):
            Whether to print information about the directory creation. Default: False.

        clear (bool, optional):
            Whether to clear the directory if it already exists. Default: False.
    """
    try:
        if isinstance(path, str):
            if len(path) > 0:
                os.makedirs(path)
        else:
            print("create_dir, ERROR: path is no str")
    except Exception:
        if os.path.isdir(path):
            if print_info:
                print(path + " already exists")
            if clear:
                ### clear folder
                ### do you really want?
                answer = input(f"Do you really want to clear {path} (y/n):")
                while answer != "y" and answer != "n":
                    print("please enter y or n")
                    answer = input(f"Do you really want to clear {path} (y/n):")
                ### clear or not depending on answer
                if answer == "y":
                    clear_dir(path)
                    if print_info:
                        print(path + " already exists and was cleared.")
                else:
                    if print_info:
                        print(path + " already exists and was not cleared.")
        else:
            print(traceback.format_exc())
            print("could not create " + path + " folder")
            quit()


def save_variables(variable_list: list, name_list: list, path: str | list = "./"):
    """
    Args:
        variable_list (list):
            variables to save
        name_list (list):
            names of the save files of the variables
        path (str or list):
            save path for all variables, or save path for each variable of the
            variable_list. Default: "./"

    Examples:
        ```python
        import numpy as np
        from CompNeuroPy import save_variables, load_variables

        ### create variables
        var1 = np.random.rand(10)
        var2 = np.random.rand(10)

        ### save variables
        save_variables([var1, var2], ["var1_file", "var2_file"], "my_variables_folder")

        ### load variables
        loaded_variables = load_variables(["var1", "var2"], "my_variables_folder")

        ### use loaded variables
        print(loaded_variables["var1_file"])
        print(loaded_variables["var2_file"])
        ```
    """
    for idx in range(len(variable_list)):
        ### set save path
        if isinstance(path, str):
            save_path = path
        else:
            save_path = path[idx]
        if save_path.endswith("/"):
            save_path = save_path[:-1]
        ### set file name
        file_name = f"{name_list[idx]}.pkl"
        ### set variable
        variable = variable_list[idx]
        ### generate save folder
        create_dir(save_path)
        ### Saving a variable to a file
        with open(f"{save_path}/{file_name}", "wb") as file:
            pickle.dump(variable, file)


def load_variables(name_list: list, path: str | list = "./"):
    """
    Args:
        name_list (list):
            names of the save files of the variables
        path (str or list, optional):
            save path for all variables, or save path for each variable of the
            variable_list. Default: "./"

    Returns:
        variable_dict (dict):
            dictionary with the loaded variables, keys are the names of the
            files, values are the loaded variables

    Examples:
        ```python
        import numpy as np
        from CompNeuroPy import save_variables, load_variables

        ### create variables
        var1 = np.random.rand(10)
        var2 = np.random.rand(10)

        ### save variables
        save_variables([var1, var2], ["var1_file", "var2_file"], "my_variables_folder")

        ### load variables
        loaded_variables = load_variables(["var1", "var2"], "my_variables_folder")

        ### use loaded variables
        print(loaded_variables["var1_file"])
        print(loaded_variables["var2_file"])
        ```
    """
    variable_dict = {}
    for idx in range(len(name_list)):
        ### set save path
        if isinstance(path, str):
            save_path = path
        else:
            save_path = path[idx]
        if save_path.endswith("/"):
            save_path = save_path[:-1]
        ### set file name
        file_name = f"{name_list[idx]}.pkl"
        ### Loading the variable from the file
        with open(f"{save_path}/{file_name}", "rb") as file:
            loaded_variable = pickle.load(file)
        ### store variable in variable_dict
        variable_dict[name_list[idx]] = loaded_variable

    return variable_dict


def timing_decorator(threshold=0.1):
    """
    Decorator to measure the execution time of a function.

    Args:
        threshold (float, optional):
            Threshold in seconds. If the execution time of the function is
            larger than this threshold, the execution time is printed. Default: 0.1.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time()
            result = func(*args, **kwargs)
            end_time = time()
            execution_time = end_time - start_time
            if execution_time >= threshold:
                print(f"{func.__name__} took {execution_time:.4f} seconds")
            return result

        return wrapper

    return decorator


def run_script_parallel(
    script_path: str, n_jobs: int, args_list: list = [""], n_total: int = 1
):
    """
    Run a script in parallel.

    Args:
        script_path (str):
            Path to the script to run.
        n_jobs (int):
            Number of parallel jobs.
        args_list (list, optional):
            List of lists containing the arguments (string values) of each run to pass
            to the script. Length of the list is the number of total runs. If a list
            of strings is passed these arguments are passed to the script and it is run
            n_total times. Default: [""], i.e. no arguments are passed to the script.
        n_total (int, optional):
            Number of total runs, only used if args_list is not a list of lists.
            Default: 1.
    """
    ### check if args_list is a list of lists
    if not isinstance(args_list[0], list):
        args_list = [args_list] * n_total
    elif n_total != 1:
        print(
            "run_script_parallel; Warning: n_total is ignored because args_list is a list of lists"
        )

    ### do not use more jobs than necessary
    n_jobs = min(n_jobs, len(args_list))

    ### run the script in parallel
    Parallel(n_jobs=n_jobs)(
        delayed(os.system)(f"python {script_path} {' '.join(args)}")
        for args in args_list
    )


def _is_git_repo():
    try:
        # Check if the current directory is within a git repository
        subprocess.check_output(
            ["git", "rev-parse", "--is-inside-work-tree"], stderr=subprocess.STDOUT
        )
        return True
    except subprocess.CalledProcessError:
        return False


def create_data_raw_folder(
    folder_name: str,
    **kwargs,
):
    """
    Create a folder for raw data of some kind of experiments/study etc.
    All data raw should be created by RUNNING A SINGLE PYTHON script. This data should
    be stored in the folder created here.
    If the created raw data depends on some parameters, these parameters should also be
    stored. They should be global in the corresponding python script to be able to easily
    set them again (replicate the data raw creation process).
    Best practice for the python script: define global parameters at the beginning, then
    call this function.
    This function stores the following information in a file called "__data_raw_meta__"
    in the created folder:
        - the name of the python script which created the data raw
        - the global variables of the python script given as kwargs
        - the conda environment
        - the pip requirements
        - the git log of ANNarchy and CompNeuroPy if they are installed locally

    !!! warning
        Only works in a conda environment.

    Args:
        folder_name (str):
            Name of the folder to create.

        **kwargs (Any, optional):
            Global variables of the caller script.

    Examples:
        ```python
        from CompNeuroPy import create_data_raw_folder

        ### define global variables
        var1 = 1
        var2 = "test"
        var3 = [1, 2, 3]

        ### call the function
        create_data_raw_folder(
            "my_data_raw_folder",
            var1=var1,
            var2=var2,
            var3=var3,
        )
        ```
    """
    ### check if folder already exists
    if os.path.isdir(folder_name):
        raise FileExistsError(f"{folder_name} already exists")
    ### create folder
    create_dir(folder_name)

    ### get caller script
    caller_frame = inspect.stack()[1]
    caller_script = caller_frame.filename
    current_path = os.getcwd()

    ### check if in current path there is a git repository, if yes, get the current
    ### commit
    if _is_git_repo():
        ### get git log
        os.system("git log > __git_log__.txt")
        with open("__git_log__.txt", "r") as f:
            git_log = f.readlines()
        os.remove("__git_log__.txt")
        if len(git_log) == 0:
            git_log = None
        ### get git top level
        os.system("basename $(git rev-parse --show-toplevel) > __git_top__.txt")
        with open("__git_top__.txt", "r") as f:
            git_top = f.readlines()
        os.remove("__git_top__.txt")
        if len(git_top) == 0:
            git_top = None
        ### get git remote
        os.system("git remote get-url origin > __git_remote__.txt")
        with open("__git_remote__.txt", "r") as f:
            git_remote = f.readlines()
        os.remove("__git_remote__.txt")
        if len(git_remote) == 0:
            git_remote = None
    else:
        git_log = None
        git_top = None
        git_remote = None

    ### now get info for annarchy and compneuropy
    ### check with pip list if annarchy and compneuropy are editable (i.e. installed
    ### from local path with "pip install -e .")
    os.system("pip list > __pip_list__.txt")
    with open("__pip_list__.txt", "r") as f:
        pip_list = f.readlines()
    os.remove("__pip_list__.txt")
    annarchy_found = False
    compneuropy_found = False
    annarchy_found_path = ""
    compneuropy_found_path = ""
    for line in pip_list:
        if "ANNarchy" in line:
            if "/" in line:
                annarchy_found = True
                annarchy_found_path = line.split(" ")[-1]
        if "CompNeuroPy" in line:
            if "/" in line:
                compneuropy_found = True
                compneuropy_found_path = line.split(" ")[-1]

    ### if both are editable we have the paths, else check pip freeze for "@ file" (i.e.
    ### installed from local path with "pip install .")
    if not annarchy_found or not compneuropy_found:
        os.system("pip freeze > __pip_freeze__.txt")
        with open("__pip_freeze__.txt", "r") as f:
            pip_freeze = f.readlines()
        os.remove("__pip_freeze__.txt")
        for line in pip_freeze:
            if "ANNarchy" in line and "@ file://" in line and not annarchy_found:
                annarchy_found = True
                annarchy_found_path = line.split("@ file://")[-1]
            if "CompNeuroPy" in line and "@ file://" in line and not compneuropy_found:
                compneuropy_found = True
                compneuropy_found_path = line.split("@ file://")[-1]

    ### remove "\n" from paths
    annarchy_found_path = annarchy_found_path.replace("\n", "")
    compneuropy_found_path = compneuropy_found_path.replace("\n", "")

    ### if they were found get the git log of the found path
    if annarchy_found:
        ### run the following commands in the terminal, wait between the commands
        command_list = [
            "git log > __annarchy_git_log__.txt",
            f"mv __annarchy_git_log__.txt {current_path}",
        ]
        os.chdir(annarchy_found_path)
        for command in command_list:
            process = subprocess.Popen(
                command,
                shell=True,
            )
            process.wait()
        os.chdir(current_path)
        ### read the git log
        with open("__annarchy_git_log__.txt", "r") as f:
            annarchy_git_log = f.readlines()
        os.remove("__annarchy_git_log__.txt")
    if compneuropy_found:
        ### run the following commands in the terminal, wait between the commands
        command_list = [
            "git log > __compneuropy_git_log__.txt",
            f"mv __compneuropy_git_log__.txt {current_path}",
        ]
        os.chdir(compneuropy_found_path)
        for command in command_list:
            process = subprocess.Popen(
                command,
                shell=True,
            )
            process.wait()
        os.chdir(current_path)
        ### read the git log
        with open("__compneuropy_git_log__.txt", "r") as f:
            compneuropy_git_log = f.readlines()
        os.remove("__compneuropy_git_log__.txt")

    ### now get conda env file and pip requirements file to be able to replicate the
    ### environment
    ### run "conda list --explicit > conda_env.txt"
    os.system("conda list --explicit > __conda_env__.txt")
    ### run "pip-chill --no-chill > requirements.txt"
    os.system("pip-chill --no-chill > __requirements__.txt")

    ### read conda env and requirements
    with open("__conda_env__.txt", "r") as f:
        conda_env = f.readlines()
    os.remove("__conda_env__.txt")
    with open("__requirements__.txt", "r") as f:
        requirements = f.readlines()
    os.remove("__requirements__.txt")

    ### remove the line for annarchy and compneuropy from requirements if found earlier
    ### in requirements they are written with small letters
    if annarchy_found:
        requirements = [line for line in requirements if "annarchy" not in line]
    if compneuropy_found:
        requirements = [line for line in requirements if "compneuropy" not in line]

    ### store everything in a meta file
    with open(f"{folder_name}/__data_raw_meta__", "w") as f:
        git_strings = []
        if git_top:
            git_strings.append("#  " + git_top[0])
        if git_remote:
            git_strings.append("#  " + git_remote[0])
        if git_log:
            git_strings.append("#  " + git_log[0])
        f.write(
            f"# Data created by runnning\n"
            f"#  {caller_script}\n"
            f"# part of git repo:\n"
            f"{''.join(git_strings)}"
            f"# with the following global variables:\n"
        )
        for key, value in kwargs.items():
            if isinstance(value, str):
                f.write(f"{key} = '{value}'\n")
            else:
                f.write(f"{key} = {value}\n")
        f.write("\n")
        f.write(
            "# ##########################################################################\n"
        )
        f.write(
            "# START OF CONDA ENV FILE ##################################################\n"
        )
        f.write(
            "# COPY AND STORE IT AS TXT FILE ############################################\n"
        )
        for line in conda_env:
            f.write(line)
        f.write("\n")
        f.write(
            "# ##########################################################################\n"
        )
        f.write(
            "# START OF PIP REQUIREMENTS FILE ###########################################\n"
        )
        f.write(
            "# COPY AND STORE IT AS TXT FILE ############################################\n"
        )
        f.write("# This file may be used to install the python packages:\n")
        f.write("# $ pip install -r <this file>\n")
        for line in requirements:
            f.write(line)
        if annarchy_found:
            f.write("\n")
            f.write("# ANNarchy was installed locally with commit:\n")
            annarchy_commit = annarchy_git_log[0].replace("\n", "")
            f.write(f"# {annarchy_commit}")
        if compneuropy_found:
            f.write("\n")
            f.write("# CompNeuroPy was installed locally with commit:\n")
            compneuropy_commit = compneuropy_git_log[0].replace("\n", "")
            f.write(f"# {compneuropy_commit}")
