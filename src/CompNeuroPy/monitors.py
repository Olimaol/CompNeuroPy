from CompNeuroPy import analysis_functions as af
from CompNeuroPy import model_functions as mf
from ANNarchy import (
    get_time,
    reset,
    dt,
    Monitor,
    get_population,
    get_projection,
    populations,
    projections,
)
import numpy as np
from typingchecker import check_types


class CompNeuroMonitors:
    """
    Class to bring together ANNarchy monitors into one object.
    """

    def __init__(self, mon_dict={}):
        """
        Initialize CompNeuroMonitors object by creating ANNarchy monitors.

        Args:
            mon_dict (dict):
                dict with key="compartment_name;period" where period is optional and
                val=list with variables to record.
        """
        self.mon = self._add_monitors(mon_dict)
        self.mon_dict = mon_dict
        self._init_internals(init_call=True)

    def _init_internals(self, init_call=False):
        """
        Initialize the following internal variables:
            - timings (dict):
                dict with key="pop_name" for populations and "proj_name" for projections
                for each recorded population and projection and
                val={"currently_paused": True, "start": [], "stop": []}
            - recordings (list):
                list with recordings of all chunks. Set to empty list.
            - recording_times (list):
                list with recording times of all chunks. Set to empty list.
            - already_got_recordings (bool):
                True if recordings were already requested, False otherwise. Set to
                False.
            - already_got_recording_times (bool):
                True if recording_times were already requested, False otherwise. Set to
                False.
            - get_recordings_reset_call (bool):
                True if get_recordings() and get_recording_times() are called within
                reset(), False otherwise. Set to False.

        Args:
            init_call (bool, optional):
                True if called from __init__(), False otherwise. Default: False.
        """
        if init_call is False:
            #### pause all ANNarchy monitors because currently paused will be set to False
            self.pause()

        ### initialize timings
        timings = {}
        for key, val in self.mon_dict.items():
            _, compartment, _ = self._unpack_mon_dict_keys(key)
            timings[compartment] = {"currently_paused": True, "start": [], "stop": []}
        self.timings = timings

        ### initialize recordings and recording_times etc.
        self.recordings = []
        self.recording_times = []
        self.already_got_recordings = False
        self.already_got_recording_times = False
        self.get_recordings_reset_call = False

    @check_types()
    def start(self, compartment_list: list | None = None):
        """
        Start or resume recording of all recorded compartments in compartment_list.

        Args:
            compartment_list (list, optional):
                List with compartment names to start or resume recording. Default: None,
                i.e., all compartments of initialized mon_dict are started or resumed.
        """
        if compartment_list == None:
            mon_dict_key_list = list(self.mon_dict.keys())
            compartment_list = [
                self._unpack_mon_dict_keys(key)[1] for key in mon_dict_key_list
            ]

        self.timings = self._start_monitors(compartment_list, self.mon, self.timings)

    @check_types()
    def pause(self, compartment_list: list | None = None):
        """
        Pause recording of all recorded compartments in compartment_list.

        Args:
            compartment_list (list, optional):
                List with compartment names to pause recording. Default: None,
                i.e., all compartments of initialized mon_dict are paused.
        """
        if compartment_list == None:
            mon_dict_key_list = list(self.mon_dict.keys())
            compartment_list = [
                self._unpack_mon_dict_keys(key)[1] for key in mon_dict_key_list
            ]

        self.timings = self._pause_monitors(compartment_list, self.mon, self.timings)

    def reset(
        self,
        populations=True,
        projections=False,
        synapses=False,
        monitors=True,
        model=True,
        parameters=True,
        net_id=0,
    ):
        """
        Create a new recording chunk by getting recordings and recording times of the
        current chunk and optionally resetting the model. Recordings are automatically
        resumed in the new chunk if they are not paused.

        Args:
            populations (bool, optional):
                If True, reset populations. Default: True.
            projections (bool, optional):
                If True, reset projections. Default: False.
            synapses (bool, optional):
                If True, reset synapses. Default: False.
            monitors (bool, optional):
                If True, reset ANNarchy monitors. Default: True.
            model (bool, optional):
                If True, reset model. Default: True.
            parameters (bool, optional):
                If True, reset the parameters of popilations and projections. Default:
                True.
            net_id (int, optional):
                Id of the network to reset. Default: 0.
        """
        ### TODO rename this function to new_chunk() or something like that and let
        ### recordings and recording times be returned
        self.get_recordings_reset_call = True
        self.get_recordings()
        self.get_recording_times()
        self.get_recordings_reset_call = False
        self.already_got_recordings = (
            False  # after reset one can still update recordings
        )
        self.already_got_recording_times = (
            False  # after reset one can still update recording_times
        )

        ### reset timings, after reset, add a zero to start if the monitor is still
        ### running (this is not resetted by reset())
        ### if the model was not resetted --> do add current time instead of zero
        for key in self.timings.keys():
            self.timings[key]["start"] = []
            self.timings[key]["stop"] = []
            if self.timings[key]["currently_paused"] == False:
                if model:
                    self.timings[key]["start"].append(0)
                else:
                    self.timings[key]["start"].append(
                        np.round(get_time(), af.get_number_of_decimals(dt()))
                    )

        ### reset model
        if model:
            if parameters is False:
                ### if parameters=False, get parameters before reset and set them after
                ### reset
                parameters_dict = mf._get_all_parameters()
            reset(populations, projections, synapses, monitors, net_id=net_id)
            if parameters is False:
                ### if parameters=False, set parameters after reset
                mf._set_all_parameters(parameters_dict)

    def current_chunk(self):
        """
        Get the index of the current chunk.

        Returns:
            current_chunk_idx (int):
                Index of the current chunk. If no recordings are currently active,
                returns None.
        """
        ### if recordings are currently active --> return chunk in which these recordings will be saved
        ### check if there are currently active recordings
        active_recordings = False
        for key, val in self.mon_dict.items():
            _, compartment, _ = self._unpack_mon_dict_keys(key)
            if not (self.timings[compartment]["currently_paused"]):
                ### tere are currently active recordings
                active_recordings = True

        if active_recordings:
            current_chunk_idx = len(self.recordings)
            return current_chunk_idx
        else:
            ### if currently no recordings are active return None
            return None

    def get_recordings(self) -> list[dict]:
        """
        Get recordings of all recorded compartments.

        Returns:
            recordings (list):
                List with recordings of all chunks.
        """
        ### only if recordings in current chunk and get_recodings was not already called add current chunk to recordings
        if (
            self._any_recordings_in_current_chunk()
            and self.already_got_recordings is False
        ):
            ### update recordings
            self.recordings.append(self._get_monitors(self.mon_dict, self.mon))
            ### upade already_got_recordings --> it will not update recordings again
            self.already_got_recordings = True

            if not (self.get_recordings_reset_call):
                if len(self.recordings) == 0:
                    print(
                        "WARNING get_recordings: no recordings available, empty list returned. Maybe forgot start()?"
                    )
            return self.recordings
        else:
            if not (self.get_recordings_reset_call):
                if len(self.recordings) == 0:
                    print(
                        "WARNING get_recordings: no recordings available, empty list returned. Maybe forgot start()?"
                    )
            return self.recordings

    def get_recording_times(self):
        """
        Get recording times of all recorded compartments.

        Returns:
            recording_times (recording_times_cl):
                Object with recording times of all chunks.
        """

        temp_timings = self._get_temp_timings()

        ### only append temp_timings of current chunk if there are recordings in current chunk at all and if get_recordings was not already called (double call would add the same chunk again)
        if (
            self._any_recordings_in_current_chunk()
            and self.already_got_recording_times is False
        ):
            self.recording_times.append(temp_timings)

        ### upade already_got_recording_times --> it will not update recording_times again
        self.already_got_recording_times = True

        ### generate a object from recording_times and return this instead of the dict
        recording_times_ob = RecordingTimes(self.recording_times)

        if not (self.get_recordings_reset_call):
            if len(self.recording_times) == 0:
                print(
                    "WARNING get_recording_times: no recordings available, empty list returned. Maybe forgot start()?"
                )
        return recording_times_ob

    def get_recordings_and_clear(self):
        """
        The default get_recordings method should be called at the end of the simulation.
        The get_recordings_and_clear method allows to get several times recordings with
        the same monitor object and to simulate between the calls. Sets the internal
        variables back to their initial state. Usefull if you repeat a simulation +
        recording several times and you do not want to always create new chunks.

        !!! warning
            If you want to continue recording after calling this method, you have to
            call start() again.

        Returns:
            recordings (list):
                List with recordings of all chunks.
            recording_times (recording_times_cl):
                Object with recording times of all chunks.
        """
        ret0 = self.get_recordings()
        ret1 = self.get_recording_times()
        self._init_internals()
        ret = (ret0, ret1)
        return ret

    def _correct_start_stop(self, start_time_arr, stop_time_arr, period):
        """
        Corrects the start and stop times of recordings to the actual start and stop
        times of recorded values.

        Args:
            start_time_arr (np.array):
                Array with start times of recordings, obtained with get_time() function
                of ANNarchy.
            stop_time_arr (np.array):
                Array with stop times of recordings, obtained with get_time() function
                of ANNarchy.
            period (float):
                Time difference between recording values specified by the user.

        Returns:
            actual_start_time (np.array):
                Array with actual start times of recorded values.
            actual_stop_time (np.array):
                Array with actual stop times of recorded values.
            nr_rec_vals (np.array):
                Array with number of recorded values between start and stop.
        """
        # actual_period = int(period / dt()) * dt()
        actual_start_time = np.ceil(start_time_arr / period) * period

        actual_stop_time = np.ceil(stop_time_arr / period - 1) * period

        nr_rec_vals = 1 + (actual_stop_time - actual_start_time) / period

        return (actual_start_time, actual_stop_time, nr_rec_vals)

    def _get_temp_timings(self):
        """
        Generates a timings dictionary with time lims and idx lims for each compartment.
        Calculates the idx lims of the recordings based on the time lims.

        Returns:
            temp_timings (dict):
                Dict with time lims and idx lims for each compartment.
        """
        temp_timings = {}
        for key in self.mon_dict.keys():
            _, compartment, period = self._unpack_mon_dict_keys(key)
            if len(self.timings[compartment]["start"]) > len(
                self.timings[compartment]["stop"]
            ):
                ### was started/resumed but never stoped after --> use current time for stop time
                self.timings[compartment]["stop"].append(get_time())
            ### calculate the idx of the recorded arrays which correspond to the timings and remove 'currently_paused'
            ### get for each start-stop pair the corrected start stop timings (when teh values were actually recorded, depends on period and timestep)
            ### and also get the number of recorded values for start-stop pair
            start_time_arr = np.array(self.timings[compartment]["start"])
            stop_time_arr = np.array(self.timings[compartment]["stop"])
            (
                start_time_arr,
                stop_time_arr,
                nr_rec_vals_arr,
            ) = self._correct_start_stop(start_time_arr, stop_time_arr, period)

            ### with the number of recorded values -> get start and end idx for each start-stop pair
            start_idx = [
                np.sum(nr_rec_vals_arr[0:i]).astype(int)
                for i in range(nr_rec_vals_arr.size)
            ]
            stop_idx = [
                np.sum(nr_rec_vals_arr[0 : i + 1]).astype(int) - 1
                for i in range(nr_rec_vals_arr.size)
            ]

            ### return start-stop pair info in timings format
            temp_timings[compartment] = {
                "start": {
                    "ms": np.round(
                        start_time_arr, af.get_number_of_decimals(dt())
                    ).tolist(),
                    "idx": start_idx,
                },
                "stop": {
                    "ms": np.round(
                        stop_time_arr, af.get_number_of_decimals(dt())
                    ).tolist(),
                    "idx": stop_idx,
                },
            }
        return temp_timings

    def _any_recordings_in_current_chunk(self):
        """
        Check if there are any recordings in the current chunk.

        Returns:
            any_recordings (bool):
                True if there are any recordings in the current chunk, False otherwise.
        """
        temp_timings = self._get_temp_timings()

        ### generate a temp object of temp timings to check if there were recordings at all
        recording_times_ob_temp = RecordingTimes([temp_timings])
        return recording_times_ob_temp._any_recordings(chunk=0)

    def _add_monitors(self, mon_dict: dict):
        """
        Generate monitors defined by mon_dict.

        Args:
            mon_dict (dict):
                dict with key="compartment_name;period" where period is optional and
                val=list with variables to record.

        Returns:
            mon (dict):
                dict with key="pop_name" for populations and key="proj_name" for
                projections and val=ANNarchy monitor object.
        """
        mon = {}
        for key, val in mon_dict.items():
            compartmentType, compartment, period = self._unpack_mon_dict_keys(
                key, warning=True
            )
            ### check if compartment is pop
            if compartmentType == "pop":
                mon[compartment] = Monitor(
                    get_population(compartment), val, start=False, period=period
                )
            ### check if compartment is proj
            if compartmentType == "proj":
                mon[compartment] = Monitor(
                    get_projection(compartment), val, start=False, period=period
                )
        return mon

    def _start_monitors(self, compartment_list, mon, timings=None):
        """
        Starts or resumes monitores defined by compartment_list.

        Args:
            compartment_list (list):
                List with compartment names to start or resume recording.
            mon (dict):
                Dict with key="pop_name" for populations and key="proj_name" for
                projections and val=ANNarchy monitor object.
            timings (dict, optional):
                timings variable of the CompNeuroMonitors object. Default: None.

        Returns:
            timings (dict):
                timings variable of the CompNeuroMonitors object.
        """
        ### for each compartment generate started variable (because compartments can ocure multiple times if multiple variables of them are recorded --> do not start same monitor multiple times)
        started = {}
        for compartment_name in compartment_list:
            started[compartment_name] = False

        if timings == None:
            ### information about pauses not available, just start
            for compartment_name in compartment_list:
                if started[compartment_name] == False:
                    mon[compartment_name].start()
                    print("start", compartment_name)
                    started[compartment_name] = True
            return None
        else:
            ### information about pauses available, start if not paused, resume if paused
            for compartment_name in compartment_list:
                if started[compartment_name] == False:
                    if timings[compartment_name]["currently_paused"]:
                        if len(timings[compartment_name]["start"]) > 0:
                            ### resume
                            mon[compartment_name].resume()
                        else:
                            ### initial start
                            mon[compartment_name].start()
                    started[compartment_name] = True
                    ### update currently_paused
                    timings[compartment_name]["currently_paused"] = False
                    ### never make start longer than stop+1!... this can be caused if start is called multiple times without pause in between
                    if len(timings[compartment_name]["start"]) <= len(
                        timings[compartment_name]["stop"]
                    ):
                        timings[compartment_name]["start"].append(get_time())
            return timings

    def _pause_monitors(self, compartment_list, mon, timings=None):
        """
        Pause monitores defined by compartment_list.

        Args:
            compartment_list (list):
                List with compartment names to pause recording.
            mon (dict):
                Dict with key="pop_name" for populations and key="proj_name" for
                projections and val=ANNarchy monitor object.
            timings (dict, optional):
                timings variable of the CompNeuroMonitors object. Default: None.

        Returns:
            timings (dict):
                timings variable of the CompNeuroMonitors object.
        """
        ### for each compartment generate paused variable (because compartments can ocure multiple times if multiple variables of them are recorded --> do not pause same monitor multiple times)
        paused = {}
        for compartment_name in compartment_list:
            paused[compartment_name] = False

        for compartment_name in compartment_list:
            if paused[compartment_name] == False:
                mon[compartment_name].pause()
                paused[compartment_name] = True

        if timings != None:
            ### information about pauses is available, update it
            for key, val in paused.items():
                timings[key]["currently_paused"] = True
                ### never make pause longer than start, this can be caused if pause is called multiple times without start in between
                if len(timings[key]["stop"]) < len(timings[key]["start"]):
                    timings[key]["stop"].append(get_time())
                ### if pause is directly called after start --> start == stop --> remove these entries, this is no actual period
                if (
                    len(timings[key]["stop"]) == len(timings[key]["start"])
                    and timings[key]["stop"][-1] == timings[key]["start"][-1]
                ):
                    timings[key]["stop"] = timings[key]["stop"][:-1]
                    timings[key]["start"] = timings[key]["start"][:-1]
            return timings
        else:
            return None

    def _get_monitors(self, mon_dict, mon):
        """
        Get recorded values from ANNarchy monitors defined by mon_dict.

        Args:
            mon_dict (dict):
                dict with key="compartment_name;period" where period is optional and
                val=list with variables to record.
            mon (dict):
                Dict with key="pop_name" for populations and key="proj_name" for
                projections and val=ANNarchy monitor object.

        Returns:
            recordings (dict):
                Dict with key="compartment_name;variable" and val=list with recorded
                values.
        """
        recordings = {}
        for key, val in mon_dict.items():
            compartment_type, compartment, period = self._unpack_mon_dict_keys(key)
            recordings[f"{compartment};period"] = period
            if compartment_type == "pop":
                pop = get_population(compartment)
                parameter_dict = {
                    param_name: getattr(pop, param_name)
                    for param_name in pop.parameters
                }
                recordings[f"{compartment};parameter_dict"] = parameter_dict
            if compartment_type == "proj":
                proj = get_projection(compartment)
                parameter_dict = {
                    param_name: getattr(proj, param_name)
                    for param_name in proj.parameters
                }
                recordings[f"{compartment};parameters"] = parameter_dict
            for val_val in val:
                temp = mon[compartment].get(val_val)
                recordings[f"{compartment};{val_val}"] = temp
        recordings["dt"] = dt()
        return recordings

    def _unpack_mon_dict_keys(self, s: str, warning: bool = False):
        """
        Unpacks a string of the form "compartment_name;period" or
        "compartment_name" into its components. If period is not provided
        it is set to dt() for populations and dt()*1000 for projections.

        Args:
            s (str):
                String to be unpacked
            warning (bool, optional):
                If True, print warning if period is not provided for projections.

        Returns:
            compartment_type (str):
                Compartment type
            compartment_name (str):
                Compartment name
            period (float):
                Period of the compartment
        """
        ### split string
        splitted_s = s.split(";")

        ### get name
        compartment_name = splitted_s[0]

        ### get type
        pop_list = [pop.name for pop in populations()]
        proj_list = [proj.name for proj in projections()]
        if compartment_name in pop_list and compartment_name in proj_list:
            ### raise error because name is in both lists
            print(
                "ERROR CompNeuroMonitors._unpack_mon_dict_keys(): compartment_name is both populaiton and projection"
            )
            quit()
        elif compartment_name in pop_list:
            compartment_type = "pop"
        elif compartment_name in proj_list:
            compartment_type = "proj"

        ### get period
        if len(splitted_s) == 2:
            period = float(splitted_s[1])
        else:
            period = {"pop": dt(), "proj": dt() * 1000}[compartment_type]
            ### print warning for compartment_type proj
            if compartment_type == "proj" and warning:
                print(
                    f"WARNING CompNeuroMonitors: no period provided for projection {compartment_name}, period set to {period} ms"
                )
        period = round(period / dt()) * dt()

        return compartment_type, compartment_name, period


### old name for backwards compatibility, TODO: remove in future
Monitors = CompNeuroMonitors


class RecordingTimes:
    def __init__(self, recording_times_list):
        """
        Initialize RecordingTimes object.

        Args:
            recording_times_list (list):
                List with recording times of all chunks.
        """
        self.recording_times_list = recording_times_list

    def time_lims(
        self,
        chunk: int | None = None,
        compartment: str | None = None,
        period: int | None = None,
    ):
        """
        Get the time limits recordings of of a specified chunk/model compartment in ms.

        chunk (int, optional):
            Index of the chunk. Default: None, i.e., first chunk.
        compartment (str, optional):
            Name of the compartment. Default: None, i.e., first model compartment from
            monitor.
        period (int, optional):
            Index of the period. Default: None, i.e., all periods.

        Returns:
            lims (tuple):
                Tuple with start and stop time of the specified chunk/model compartment.
        """
        assert (
            len(self.recording_times_list) > 0
        ), "ERROR time_lims(): No recordings/recording_times available."
        return self._lims("ms", chunk, compartment, period)

    def idx_lims(
        self,
        chunk: int | None = None,
        compartment: str | None = None,
        period: int | None = None,
    ):
        """
        Get the index limits of recordings of a specified chunk/model compartment.

        chunk (int, optional):
            Index of the chunk. Default: None, i.e., first chunk.
        compartment (str, optional):
            Name of the compartment. Default: None, i.e., first model compartment from
            monitor.
        period (int, optional):
            Index of the period. Default: None, i.e., all periods.

        Returns:
            lims (tuple):
                Tuple with start and stop index of the specified chunk/model
                compartment.
        """
        assert (
            len(self.recording_times_list) > 0
        ), "ERROR idx_lims(): No recordings/recording_times available."
        return self._lims("idx", chunk, compartment, period)

    def all(self):
        """
        Get the recording times of all chunks, compartments, periods in ms and index.

        Returns:
            recording_times_list (list):
                List with recording times of all chunks.
        """
        return self.recording_times_list

    def nr_periods(self, chunk=None, compartment=None):
        """
        Get the number of recording periods (start-pause) of a specified chunk/model
        compartment.

        Args:
            chunk (int, optional):
                Index of the chunk. Default: None, i.e., first chunk.
            compartment (str, optional):
                Name of the compartment. Default: None, i.e., first model compartment
                from monitor.

        Returns:
            nr_periods (int):
                Number of recording periods (start-pause) of a specified chunk/model
                compartment.
        """
        chunk = self._check_chunk(chunk)
        compartment = self.__check_compartment__(compartment, chunk)
        return self._get_nr_periods(chunk, compartment)

    def combine_chunks(
        self, recordings: list, recording_data_str: str, mode="sequential"
    ):
        """
        Combines the data of all chunks of recordings, only possible if no pauses in
        between.

        Args:
            recordings (list):
                List with recordings of all chunks.
            recording_data_str (str):
                String specifying the compartment name and the variable to combine.
                Format: "compartment_name;variable_name"
            mode (str, optional):
                How should the time array be generated. Can be "sequential" or
                "consecutive". Default: "sequential".
                - "sequential": each chunk starts at zero e.g.: [0,100] + [0,250] -->
                    [0, 1, ..., 100, 0, 1, ..., 250]
                - "consecutive": each chunk starts at the last stop time of the previous
                    chunk e.g.: [0,100] + [0,250] --> [0, 1, ..., 100, 101, 102, ..., 350]

        Returns:
            time_arr (np.array):
                Array with time values in ms.
            data_arr (np.array):
                Array with the recorded variable.
        """
        assert (
            len(self.recording_times_list) > 0
        ), "ERROR combine_chunks(): No recordings/recording_times available."

        compartment = recording_data_str.split(";")[0]
        period_time = recordings[0][f"{compartment};period"]
        time_step = recordings[0]["dt"]
        nr_chunks = self._get_nr_chunks()
        data_list = []
        time_list = []
        pre_chunk_start_time = 0

        for chunk in range(nr_chunks):
            ### append data list with data of all periods of this chunk
            data_list.append(recordings[chunk][recording_data_str])

            ### nr of periods in this chunk
            nr_periods = self._get_nr_periods(chunk, compartment)

            ### start time of chunk depends on mode
            if mode == "sequential":
                chunk_start_time = 0
            elif mode == "consecutive":
                if chunk == 0:
                    chunk_start_time = 0
                else:
                    last_stop_time = self.recording_times_list[chunk - 1][compartment][
                        "stop"
                    ]["ms"][-1]
                    chunk_start_time = (
                        pre_chunk_start_time + last_stop_time + period_time
                    )
                    pre_chunk_start_time = chunk_start_time
            else:
                print("ERROR recording_times.combine_data, Wrong mode.")
                quit()

            ### append the time list with all times of the periods
            for period in range(nr_periods):
                start_time = (
                    self.time_lims(chunk=chunk, compartment=compartment, period=period)[
                        0
                    ]
                    + chunk_start_time
                )
                end_time = (
                    self.time_lims(chunk=chunk, compartment=compartment, period=period)[
                        1
                    ]
                    + chunk_start_time
                )
                start_time = round(start_time, af.get_number_of_decimals(time_step))
                end_time = round(end_time, af.get_number_of_decimals(time_step))
                times = np.arange(start_time, end_time + period_time, period_time)
                time_list.append(times)

        ### flatten the two lists
        data_arr = np.concatenate(data_list, 0)
        time_arr = np.concatenate(time_list, 0)

        ### check if there are gaps in the time array
        ### fill them with the corersponding times and
        ### the data array with nan values
        time_arr, data_arr = af.time_data_add_nan(
            time_arr,
            data_arr,
            fill_time_step=period_time,
        )

        return time_arr, data_arr

    def _lims(self, string, chunk=None, compartment=None, period=None):
        """
        Get the limits of recordings of a specified chunk/model compartment.

        Args:
            string (str):
                String specifying the type of limits to return. Can be "ms" for time
                limits in ms or "idx" for index limits.
            chunk (int, optional):
                Index of the chunk. Default: None, i.e., first chunk.
            compartment (str, optional):
                Name of the compartment. Default: None, i.e., first model compartment
                from monitor.
            period (int, optional):
                Index of the period. Default: None, i.e., all periods.

        Returns:
            lims (tuple):
                Tuple with start and stop time/index of the specified chunk/model
                compartment.
        """

        chunk = self._check_chunk(chunk)
        compartment = self.__check_compartment__(compartment, chunk)
        period_0, period_1 = self._check_period(period, chunk, compartment)
        lims = (
            self.recording_times_list[chunk][compartment]["start"][string][period_0],
            self.recording_times_list[chunk][compartment]["stop"][string][period_1],
        )
        return lims

    def __check_compartment__(self, compartment, chunk):
        if compartment == None:
            ### by default just use the first compartment
            compartment = list(self.recording_times_list[chunk].keys())[0]
        elif compartment in list(self.recording_times_list[chunk].keys()):
            compartment = compartment
        else:
            print(
                'ERROR recording_times, given compartment "'
                + str(compartment)
                + '" not available'
            )
            quit()

        return compartment

    def _check_period(self, period, chunk, compartment):
        """
        Check if period is given.

        Args:
            period (int, optional):
                Index of the period. Default: None, i.e., all periods.
            chunk (int):
                Index of the chunk.
            compartment (str):
                Name of the compartment.

        Returns:
            period_0 (int):
                Index of the first period.
            period_1 (int):
                Index of the last period. If perios is given, period_0 == period_1.
        """
        if period == None:
            ### by default use all periods
            period_0 = 0
            period_1 = (
                len(self.recording_times_list[chunk][compartment]["start"]["idx"]) - 1
            )
        elif period < len(
            self.recording_times_list[chunk][compartment]["start"]["idx"]
        ):
            period_0 = period
            period_1 = period
        else:
            print("ERROR recording_times, given period not available")
            quit()

        return period_0, period_1

    def _check_chunk(self, chunk):
        """
        Check if chunk is given.

        Args:
            chunk (int, optional):
                Index of the chunk. Default: None, i.e., first chunk.

        Returns:
            chunk (int):
                Index of the chunk.
        """
        if chunk is None:
            ### by default use first chunk
            chunk = 0
        elif chunk < self._get_nr_chunks():
            chunk = chunk
        else:
            print("ERROR recording_times, given chunk not available")
            quit()

        return chunk

    def _get_nr_chunks(self):
        """
        Get the number of chunks of the recordings.

        Returns:
            nr_chunks (int):
                Number of chunks.
        """
        return len(self.recording_times_list)

    def _get_nr_periods(self, chunk, compartment):
        """
        Get the number of recording periods (start-pause) of a specified chunk/model
        compartment.

        Args:
            chunk (int):
                Index of the chunk.
            compartment (str):
                Name of the compartment.

        Returns:
            nr_periods (int):
                Number of recording periods (start-pause) of a specified chunk/model
                compartment.
        """
        return len(self.recording_times_list[chunk][compartment]["start"]["idx"])

    def _any_recordings(self, chunk):
        """
        Check all periods and compartments if there are any recordings.

        Args:
            chunk (int):
                Index of the chunk.

        Returns:
            found_recordings (bool):
                True if there are any recordings in the chunk, False otherwise.
        """
        compartment_list = list(self.recording_times_list[chunk].keys())
        found_recordings = False
        for compartment in compartment_list:
            nr_periods_of_compartment = len(
                self.recording_times_list[chunk][compartment]["start"]["idx"]
            )

            for period_idx in range(nr_periods_of_compartment):
                idx_lims = self.idx_lims(
                    chunk=chunk, compartment=compartment, period=period_idx
                )
                if np.diff(idx_lims)[0] > 0:
                    found_recordings = True

        return found_recordings


### old name for backwards compatibility, TODO: remove in future
recording_times_cl = RecordingTimes
