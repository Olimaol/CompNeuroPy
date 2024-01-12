## Create Monitors
CompNeuroPy provides a [`CompNeuroMonitors`](#CompNeuroPy.monitors.CompNeuroMonitors) class that can be used to easily create and control multiple ANNarchy monitors at once. To create a [`CompNeuroMonitors`](#CompNeuroPy.monitors.CompNeuroMonitors) object, all that is needed is a monitors_dictionary that defines which variables should be recorded for each model component. All populations and projections have to have unique names to work with [`CompNeuroMonitors`](#CompNeuroPy.monitors.CompNeuroMonitors). The keys of the monitor_dictionary are the names of the model components (in example below _"my_pop1"_ and _"my_pop2"_). The key can also include a recording period (the time between two recordings, given after a ";"), e.g. record the variables of _my_pop1_ only every 10 ms would look like this: _'pop;my_pop1;10':['v', 'spike']_. The default period is the time step of the simulation for populations and 1000 times the timestep for projections. The values of the monitor_dictionary are lists of all the variables that should be recorded from the corresponding components. The names of components (populations, projections) could be provided by a [`CompNeuroModel`](generate_models.md#CompNeuroPy.generate_model.CompNeuroModel).

### Example:
Here the variables _v_ and _spike_ should be recorded of the population with the name _"my_pop1"_ and the variable _v_ should be recorded from the population with the name _"my_pop2"_:

```python
from CompNeuroPy import CompNeuroMonitors
monitor_dictionary = {'my_pop1':['v', 'spike'], 'my_pop2':['v']}
mon = CompNeuroMonitors(monitor_dictionary)
```

A full example is available in the [Examples](../examples/monitor_recordings.md).

## Chunks and periods
In CompNeuroPy, recordings are divided into so-called chunks and periods. Chunks are simulation sections that are separated by monitor resets (optionally also reset the model). A chunk can consist of several periods. A period represents the time span between the start and pause of a monitor recording. To divide a simulation into chunks and periods, a [`CompNeuroMonitors`](#CompNeuroPy.monitors.CompNeuroMonitors) object provides the three functions _start()_, _pause()_ and _reset()_.

At the beginning of a simulation, the monitors do not start automatically which is why the _start()_ function must be called at least once. The _start()_ function can also be used to resume paused recordings. With the function _pause()_ recordings are paused. The function _reset()_ starts a new chunk for the recordings (the end of a chunk is also always the end of a period, i.e. the last period of the corresponding chunk). After calling _reset()_ the monitors remain in their current mode (active or paused). By default _reset()_ also resets the model to the compile status (time = 0) by calling the ANNarchy _reset()_ function and has the same arguments. If the argument _model_ is set to False, the ANNarchy _reset()_ function is not called and only a new chunk is created.

### Example:
```python
### first chunk, one period
simulate(100) # 100 ms not recorded
mon.start()   # start all monitors
simulate(100) # 100 ms recorded

### second chunk, two periods
mon.reset()   # model reset, beginning of new chunk
simulate(100) # 100 ms recorded (monitors were active before reset --> still active)
mon.pause()   # pause all monitors
simulate(100) # 100 ms not recorded
mon.start()   # start all monitors
simulate(100) # 100 ms recorded
```

## Get recordings
The recordings can be obtained from the [`CompNeuroMonitors`](#CompNeuroPy.monitors.CompNeuroMonitors) object using the _get_recordings()_ function. This returns a list of dictionaries (one for each chunk). The dictionaries contain the recorded data defined with the monitor_dictionary at the [`CompNeuroMonitors`](#CompNeuroPy.monitors.CompNeuroMonitors) initialization. In the recordings dictionaries the keys have the following structure: "<component_name\>;variable"; the corresponding dictionary values are the recordings of the respective variable. The dictionaries always contain the time step of the simulation (key = _"dt"_), the periods (time between recorded values) for each component (key = _"<component_name\>;period"_) and the attributes of each component (key = _"<component_name\>;parameter_dict"_).

### Example:
```python
recordings = mon.get_recordings()
y1 = recordings[0]['my_pop1;v'] ### variable v of my_pop1 from 1st chunk
y2 = recordings[1]['my_pop1;v'] ### variable v of my_pop1 from 2nd chunk
```

## Get recording times
In addition to the recordings themselves, recording times can also be obtained from the [`CompNeuroMonitors`](#CompNeuroPy.monitors.CompNeuroMonitors) object, which is very useful for later analyses. With the function _get_recording_times()_ of the [`CompNeuroMonitors`](#CompNeuroPy.monitors.CompNeuroMonitors) object a [`RecordingTimes`](#CompNeuroPy.monitors.RecordingTimes) object can be obtained. From the [`RecordingTimes`](#CompNeuroPy.monitors.RecordingTimes) object one can get time limits (in ms) and coresponding indizes for the recordings.

### Example:
```python
recording_times = mon.get_recording_times()
start_time = recording_times.time_lims(chunk=1, period=1)[0] ### 200 ms
start_idx  = recording_times.idx_lims(chunk=1, period=1)[0]  ### 1000, if dt == 0.1
end_time   = recording_times.time_lims(chunk=1, period=1)[1] ### 300 ms
end_idx    = recording_times.idx_lims(chunk=1, period=1)[1]  ### 2000
```

You can combine the recordings of both chunks of the example simulation shown above into a single time array and a single value array using the [`RecordingTimes`](#CompNeuroPy.monitors.RecordingTimes) object's combine_chunks function:
```python
time_arr, value_arr = recording_times.combine_chunks(recordings, 'my_pop1;v', 'consecutive')
```

## Plot recordings
To get a quick overview of the recordings, CompNeuroPy provides the [`PlotRecordings`](../additional/analysis_functions.md#CompNeuroPy.analysis_functions.PlotRecordings) class.


::: CompNeuroPy.monitors.CompNeuroMonitors
::: CompNeuroPy.monitors.RecordingTimes