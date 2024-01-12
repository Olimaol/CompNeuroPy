from ANNarchy import get_time
from CompNeuroPy import extra_functions as ef
from CompNeuroPy import CompNeuroMonitors
import numpy as np
from typing import Callable


class CompNeuroSim:
    """
    Class for generating a CompNeuroPy simulation.
    """

    _initialized_simulations = []

    def __init__(
        self,
        simulation_function: Callable,
        simulation_kwargs: dict | None = None,
        name: str = "simulation",
        description: str = "",
        requirements: list | None = None,
        kwargs_warning: bool = False,
        monitor_object: CompNeuroMonitors | None = None,
    ):
        """
        Args:
            simulation_function (function):
                Function which runs the simulation.
            simulation_kwargs (dict, optional):
                Dictionary of arguments for the simulation_function. Default: None.
            name (str, optional):
                Name of the simulation. Default: "simulation".
            description (str, optional):
                Description of the simulation. Default: "".
            requirements (list, optional):
                List of requirements for the simulation. It's a list of dictionaries
                which contain the requirement class itself (key: "req") and the
                corresponding arguments (keys are the names of the arguments). The
                arguments can be inherited from the simulation kwargs by using the
                syntax 'simulation_kwargs.<kwarg_name>'. Default: None.
            kwargs_warning (bool, optional):
                If True, a warning is printed if the simulation_kwargs are changed
                during the simulation. Default: False.
            monitor_object (CompNeuroMonitors object, optional):
                CompNeuroMonitors object to automatically track the recording chunk for each
                simulation run. Default: None.
        """
        # set simulation function
        self.name = name
        if name == "simulation":
            self.name = name + str(self._nr_simulations())
        self._initialized_simulations.append(self.name)
        self.description = description
        self.simulation_function = simulation_function
        self.simulation_kwargs = simulation_kwargs
        if requirements is None:
            self.requirements = []
        else:
            self.requirements = requirements
        self.start = []
        self.end = []
        self.info = []
        self.kwargs = []
        if kwargs_warning:
            self._warned = False
        else:
            self._warned = True
        self.monitor_object = monitor_object
        if monitor_object is not None:
            self.monitor_chunk = []
        else:
            self.monitor_chunk = None

        ### test initial requirements
        self._test_req(simulation_kwargs=simulation_kwargs)

    def run(self, simulation_kwargs: dict | None = None):
        """
        Runs the simulation function. With each run extend start, end list containing
        start and end time of the corresponding run and the info list containing the
        return value of the simulation function.

        Args:
            simulation_kwargs (dict, optional):
                Temporary simulation kwargs which override the initialized simulation
                kwargs. Default: None, i.e., use values from initialization.
        """

        ### define the current simulation kwargs
        if simulation_kwargs is not None:
            if self.simulation_kwargs is not None:
                ### not replace initialized kwargs completely but only the kwargs which are given
                tmp_kwargs = self.simulation_kwargs.copy()
                for key, val in simulation_kwargs.items():
                    tmp_kwargs[key] = val
            else:
                ### there are no initial kwargs --> only use the kwargs which are given
                tmp_kwargs = simulation_kwargs
            if not (self._warned) and len(self.requirements) > 0:
                print(
                    "\nWARNING! run",
                    self.name,
                    "changed simulation kwargs, initial requirements may no longer be fulfilled!\n",
                )
                self._warned = True
        else:
            tmp_kwargs = self.simulation_kwargs

        ### before each run, test requirements
        self._test_req(simulation_kwargs=tmp_kwargs)

        ### and append current simulation kwargs to the kwargs variable
        self.kwargs.append(tmp_kwargs)

        ### and append the current chunk of the monitors object to the chunk variable
        if self.monitor_object is not None:
            self.monitor_chunk.append(self.monitor_object.current_chunk())

        ### run the simulation, store start and end simulation time
        self.start.append(get_time())
        if tmp_kwargs is not None:
            self.info.append(self.simulation_function(**tmp_kwargs))
        else:
            self.info.append(self.simulation_function())
        self.end.append(get_time())

    def _nr_simulations(self):
        """
        Returns the current number of initialized CompNeuroPy simulations.
        """
        return len(self._initialized_simulations)

    def _test_req(self, simulation_kwargs=None):
        """
        Tests the initialized requirements with the current simulation_kwargs.
        """

        if simulation_kwargs is None:  # --> use the initial simulation_kwargs
            simulation_kwargs = self.simulation_kwargs

        for req in self.requirements:
            ### check if requirement_kwargs are given besides the requirement itself
            if len(list(req.keys())) > 1:
                ### remove the requirement itself from the kwargs
                req_kwargs = ef.remove_key(req, "req")
                ### check if req_kwargs reference to simulation_kwargs, if yes, use the
                ### current simulation kwargs instead of the intial ones
                for key, val in req_kwargs.items():
                    if isinstance(val, str):
                        val_split = val.split(".")
                        ### check if val is a reference to simulation_kwargs
                        if val_split[0] == "simulation_kwargs":
                            if len(val_split) == 1:
                                ### val is only simulation_kwargs
                                req_kwargs = simulation_kwargs
                            elif len(val_split) == 2:
                                ### val is simulation_kwargs.something
                                req_kwargs[key] = simulation_kwargs[val_split[1]]
                            else:
                                ### val is simulation_kwargs.something.something... e.g. key='pops' and val= 'simulation_kwargs.model.populations'
                                req_kwargs[key] = eval(
                                    'simulation_kwargs["'
                                    + val_split[1]
                                    + '"].'
                                    + ".".join(val_split[2:])
                                )
                ### run the requirement using the current req_kwargs
                req["req"](**req_kwargs).run()

            else:
                ### a requirement is given without kwargs --> just run it
                req["req"]().run()

    def get_current_arr(self, dt, flat=False):
        """
        Method exclusively for current_step simulation functions. Gets the current array
        (input current value for each time step) of all runs.

        !!! warning
            This method will be removed soon. Use the get_current_arr method of the
            SimInfo class instead.

        Args:
            dt (float):
                Time step size of the simulation.
            flat (bool, optional):
                If True, returns a flattened array. Assumes that all runs are run
                consecutively without brakes. Default: False, i.e., returns a list of
                arrays.

        Returns:
            current_arr (list of arrays):
                List of arrays containing the current values for each time step of each
                run. If flat=True, returns a flattened array.
        """
        assert (
            self.simulation_function.__name__ == "current_step"
        ), 'ERROR get_current_arr: Simulation has to be "current_step"!'
        ### TODO: remove because deprecated
        print(
            "WARNING get_current_arr function will only be available in SimInfo soon."
        )
        current_arr = []
        for run in range(len(self.kwargs)):
            t1 = self.kwargs[run]["t1"]
            t2 = self.kwargs[run]["t2"]
            a1 = self.kwargs[run]["a1"]
            a2 = self.kwargs[run]["a2"]

            if t1 > 0 and t2 > 0:
                current_arr.append(
                    np.concatenate(
                        [
                            np.ones(int(round(t1 / dt))) * a1,
                            np.ones(int(round(t2 / dt))) * a2,
                        ]
                    )
                )
            elif t2 > 0:
                current_arr.append(np.ones(int(round(t2 / dt))) * a2)
            else:
                current_arr.append(np.ones(int(round(t1 / dt))) * a1)

        if flat:
            return np.concatenate(current_arr)
        else:
            return current_arr

    def simulation_info(self):
        """
        Returns a SimInfo object containing the simulation information.

        Returns:
            simulation_info_obj (SimInfo):
                Simulation information object.
        """

        simulation_info_obj = SimInfo(
            self.name,
            self.description,
            self.simulation_function.__name__,
            self.start,
            self.end,
            self.info,
            self.kwargs,
            self.monitor_chunk,
        )

        return simulation_info_obj


### old name for backward compatibility, TODO: remove
generate_simulation = CompNeuroSim


class SimInfo:
    """
    Class for storing the simulation information.

    Attributes:
        name (str):
            Name of the simulation.
        description (str):
            Description of the simulation.
        simulation_function (str):
            Name of the simulation function.
        start (list):
            List of start times of the simulation runs.
        end (list):
            List of end times of the simulation runs.
        info (list):
            List of return values of the simulation function of each simulation run.
        kwargs (list):
            List of simulation kwargs of the simulation function of each simulation run.
        monitor_chunk (list):
            List of recording chunks of the used CompNeuroMonitors object of each simulation run.
    """

    def __init__(
        self,
        name,
        description,
        simulation_function,
        start,
        end,
        info,
        kwargs,
        monitor_chunk,
    ):
        """
        Initialization of the simulation information object.

        Args:
            name (str):
                Name of the simulation.
            description (str):
                Description of the simulation.
            simulation_function (str):
                Name of the simulation function.
            start (list):
                List of start times of the simulation runs.
            end (list):
                List of end times of the simulation runs.
            info (list):
                List of return values of the simulation function of each simulation run.
            kwargs (list):
                List of simulation kwargs of the simulation function of each simulation
                run.
            monitor_chunk (list):
                List of recording chunks of the used CompNeuroMonitors object of each simulation
                run.
        """
        self.name = name
        self.description = description
        self.simulation_function = simulation_function
        self.start = start
        self.end = end
        self.info = info
        self.kwargs = kwargs
        self.monitor_chunk = monitor_chunk

    def get_current_arr(self, dt, flat=False):
        """
        Method exclusively for the following simulation functions (built-in
        CompNeuroPy):
            - current_step
            - current_stim
            - current_ramp
        Gets the current array (input current value for each time step) of all runs.

        Args:
            dt (float):
                Time step size of the simulation.
            flat (bool, optional):
                If True, returns a flattened array. Assumes that all runs are run
                consecutively without brakes. Default: False, i.e., returns a list of
                arrays.

        Returns:
            current_arr (list of arrays):
                List of arrays containing the current values for each time step of each
                run. If flat=True, returns a flattened array.
        """
        assert (
            self.simulation_function == "current_step"
            or self.simulation_function == "current_stim"
            or self.simulation_function == "current_ramp"
        ), 'ERROR get_current_arr: Simulation has to be "current_step", "current_stim" or "current_ramp"!'

        if self.simulation_function == "current_step":
            current_arr = []
            for run in range(len(self.kwargs)):
                t1 = self.kwargs[run]["t1"]
                t2 = self.kwargs[run]["t2"]
                a1 = self.kwargs[run]["a1"]
                a2 = self.kwargs[run]["a2"]

                if t1 > 0 and t2 > 0:
                    current_arr.append(
                        np.concatenate(
                            [
                                np.ones(int(round(t1 / dt))) * a1,
                                np.ones(int(round(t2 / dt))) * a2,
                            ]
                        )
                    )
                elif t2 > 0:
                    current_arr.append(np.ones(int(round(t2 / dt))) * a2)
                else:
                    current_arr.append(np.ones(int(round(t1 / dt))) * a1)

            if flat:
                return np.concatenate(current_arr)
            else:
                return current_arr

        elif self.simulation_function == "current_stim":
            current_arr = []
            for run in range(len(self.kwargs)):
                t = self.kwargs[run]["t"]
                a = self.kwargs[run]["a"]

                if t > 0:
                    current_arr.append(np.ones(int(round(t / dt))) * a)

            if flat:
                return np.concatenate(current_arr)
            else:
                return current_arr

        elif self.simulation_function == "current_ramp":
            current_arr = []
            for run in range(len(self.kwargs)):
                amp = self.kwargs[run]["a0"]
                current_arr_ramp = []
                for stim_idx in range(self.kwargs[run]["n"]):
                    t = self.info[run]["dur_stim"]
                    a = amp
                    current_arr_ramp.append(np.ones(int(round(t / dt))) * a)
                    amp = amp + self.info[run]["da"]
                current_arr.append(list(np.concatenate(current_arr_ramp)))

            if flat:
                return np.concatenate(current_arr)
            else:
                return current_arr


### old name for backward compatibility, TODO: remove
simulation_info_cl = SimInfo
