"""
demonstrate use cases for CompNeuroMonitors

init
all paused

### 1st chunk
### demonstrate starting/pausing all
>>> start all
all started
>>> pause all
all paused
>>> start all
all started/resumed

### demonstrate pausing single compartments
>>> pause part
part paused
>>> start all
all started/resumed

### demonstrate starting single compartments
>>> pause all
all paused
>>> start part
part started/resumed
>>> start all
all started/resumed

### demonstrate chunking recordings by reset
>>> reset without model
2nd chunk, model not resetted
>>>reset with model
3rd chunk, model resetted

### demonstrate getting recordings during simulation
>>> get_recordings_and_clear
all recordings from aboth should be obtained
>>> simulate again and reset model creating 2 chunks --> check if model or ANNarchy montirs or both need to be resetted
>>> get_recordings_and_clear
new recordings (2 chunks) should be obtianed
>>> simulate again a single chunk
>>> get_recordings and get_recording_times
new recordings (1 chunk) should be obtained


"""
from ANNarchy import Population, Izhikevich, setup, simulate, get_population, compile
from CompNeuroPy import (
    generate_model,
    CompNeuroMonitors,
    create_dir,
    plot_recordings,
    PlotRecordings,
)
from CompNeuroPy.neuron_models import Izhikevich2007
import pylab as plt
import numpy as np


def main():
    ### setup ANNarchy timestep adn create results folder
    setup(dt=0.1)
    create_dir("results")

    ### first we create two populations, each consist of 1 neuron
    Population(1, neuron=Izhikevich2007(I_app=0), name="my_pop1")
    Population(1, neuron=Izhikevich2007(I_app=52), name="my_pop2")

    ### compile
    compile()

    ### after compilation we can define the monitors using the monitor_dictionary
    ### and the CompNeuroMonitors class
    ### for my_pop1 we use a recording period of 2 ms
    ### for my_pop2 we do not give a recording preiod, therefore record every timestep
    monitor_dictionary = {"my_pop1;2": ["v", "spike"], "my_pop2": ["v"]}
    mon = CompNeuroMonitors(monitor_dictionary)

    ### In this part we demonstrate starting/pausing all monitors
    ### simulate for 100 ms
    simulate(100)

    ### start all monitors
    mon.start()

    ### simulate for 100 ms
    simulate(100)

    ### pause all monitors
    mon.pause()

    ### simulate for 100 ms
    simulate(100)

    ### In this part we demonstrate starting single monitors
    ### start monitor for my_pop1
    mon.start(compartment_list=["my_pop1"])

    ### simulate for 100 ms
    simulate(100)

    ### start all monitors
    mon.start()

    ### simulate for 100 ms
    simulate(100)

    ### In this part we demonstrate pausing single monitors
    ### pause monitor for my_pop1
    mon.pause(compartment_list=["my_pop1"])

    ### simulate for 100 ms
    simulate(100)

    ### start all monitors
    mon.start()

    ### simulate for 100 ms
    simulate(100)

    ### In this part we demonstrate chunking recordings by reset
    ### reset without model, creating new chunk
    mon.reset(model=False)
    mon.pause()

    ### simulate for 100 ms
    simulate(100)

    ### start all monitors
    mon.start()

    ### simulate for 700 ms
    simulate(700)

    ### reset with model, creating new chunk
    mon.reset(model=True)
    mon.pause()

    ### simulate for 100 ms
    simulate(100)

    ### start all monitors
    mon.start()

    ### simulate for 700 ms
    simulate(700)

    ### In this part we demonstrate getting recordings during simulation
    ### get recordings using get_recordings_and_clear
    ### this also resets the model back to their initialized state, i.e. there are no
    ### recordings and they are not started
    recordings1, recording_times1 = mon.get_recordings_and_clear()
    print(mon.recordings)
    print(mon.timings, "\n")

    ### simulate for 100 ms
    simulate(100)
    print(mon.recordings)
    print(mon.timings, "\n")

    ### start all monitors
    mon.start()
    print(mon.recordings)
    print(mon.timings, "\n")

    ### simulate for 100 ms
    simulate(100)

    ### reset monitors and model
    mon.reset(model=True)

    ### simulate for 100 ms
    simulate(100)

    ### get recordings using get_recordings_and_clear
    recordings2, recording_times2 = mon.get_recordings_and_clear()

    ### reset monitors and model and start all monitors
    mon.reset(model=True)
    mon.start()

    ### simulate for 100 ms
    simulate(100)

    ### get recordings the normal way
    recordings3 = mon.get_recordings()
    recording_times3 = mon.get_recording_times()

    ### print the idx and time lims of the recordings and the sizes of the recorded
    ### arrays
    recordings_list = [recordings1, recordings2, recordings3]
    for all_times_idx, all_times in enumerate(
        [recording_times1.all(), recording_times2.all(), recording_times3.all()]
    ):
        print(f"recordings{all_times_idx+1}")
        for chunk in range(len(all_times)):
            print(f"\tchunk: {chunk}")
            for pop_name in ["my_pop1", "my_pop2"]:
                print(f"\t\tpop_name: {pop_name}")
                print(
                    f"\t\trecording_array_size: {recordings_list[all_times_idx][chunk][f'{pop_name};v'].shape}"
                )
                for time_point in ["start", "stop"]:
                    print(f"\t\t\ttime_point: {time_point}")
                    for unit in ["ms", "idx"]:
                        print(f"\t\t\t\tunit: {unit}")
                        for period in range(
                            len(all_times[chunk][pop_name][time_point][unit])
                        ):
                            print(
                                f"\t\t\t\t\tperiod {period}: {all_times[chunk][pop_name][time_point][unit][period]}"
                            )

    ### test new plot recordings
    PlotRecordings(
        figname=f"monitor_recordings_1_chunk{chunk}.png",
        recordings=recordings1,
        recording_times=recording_times1,
        chunk=0,
        shape=(2, 2),
        plan={
            "position": [1, 2, 3],
            "compartment": ["my_pop1", "my_pop2", "my_pop1"],
            "variable": ["v", "v", "spike"],
            "format": ["line", "line", "raster"],
        },
    )
    quit()

    ### plot recordings 1 consisting of 3 chunks
    for chunk in range(len(recordings1)):
        ### using plot_recordings which plots the recordings of one chunk
        plot_recordings(
            figname=f"monitor_recordings_1_chunk{chunk}.png",
            recordings=recordings1,
            recording_times=recording_times1,
            chunk=chunk,
            shape=(2, 2),
            plan=["1;my_pop1;v;line", "3;my_pop1;spike;raster", "2;my_pop2;v;line"],
        )

    ### plot recordings 2 consisting of 2 chunks
    for chunk in range(len(recordings2)):
        ### using plot_recordings which plots the recordings of one chunk
        plot_recordings(
            figname=f"monitor_recordings_2_chunk{chunk}.png",
            recordings=recordings2,
            recording_times=recording_times2,
            chunk=chunk,
            shape=(2, 2),
            plan=["1;my_pop1;v;line", "3;my_pop1;spike;raster", "2;my_pop2;v;line"],
        )

    ### plot recordings 3 consisting of 1 chunk
    for chunk in range(len(recordings3)):
        ### using plot_recordings which plots the recordings of one chunk
        plot_recordings(
            figname=f"monitor_recordings_3_chunk{chunk}.png",
            recordings=recordings3,
            recording_times=recording_times3,
            chunk=chunk,
            shape=(2, 2),
            plan=["1;my_pop1;v;line", "3;my_pop1;spike;raster", "2;my_pop2;v;line"],
        )

    return 1


if __name__ == "__main__":
    main()
