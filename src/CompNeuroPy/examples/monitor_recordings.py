"""
This example demonstrates how to use the CompNeuroMonitors class to record variables.
It is shown how to start/pause monitors, how to split recordings into chunks and
optionally reset the model and how to get recordings during and after simulation.
"""
from ANNarchy import Population, setup, simulate, compile
from CompNeuroPy import (
    CompNeuroMonitors,
    PlotRecordings,
)
from CompNeuroPy.neuron_models import Izhikevich2007


def main():
    ### setup ANNarchy timestep and create results folder
    setup(dt=0.1)

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
    ### simulate for 100 ms [0, 100]
    simulate(100)

    ### start all monitors and simulate for 100 ms [100, 200]
    mon.start()
    simulate(100)

    ### pause all monitors and simulate for 100 ms [200, 300]
    mon.pause()
    simulate(100)

    ### In this part we demonstrate starting single monitors
    ### start only monitor for my_pop1 and simulate for 100 ms [300, 400]
    mon.start(compartment_list=["my_pop1"])
    simulate(100)

    ### start all monitors and simulate for 100 ms [400, 500]
    mon.start()
    simulate(100)

    ### In this part we demonstrate pausing single monitors
    ### pause monitor for my_pop1 and simulate for 100 ms [500, 600]
    mon.pause(compartment_list=["my_pop1"])
    simulate(100)

    ### start all monitors and simulate for 100 ms [600, 700]
    mon.start()
    simulate(100)

    ### In this part we demonstrate chunking recordings by reset
    ### reset WITHOUT model, creating new chunk --> first chunk [0, 700]
    ### also in this chunk do not record the first 100 ms
    ### WITHOUT model --> time continues at 700 ms [700, 800]
    mon.reset(model=False)
    mon.pause()
    simulate(100)

    ### start all monitors and simulate for 700 ms [800, 1500]
    mon.start()
    simulate(700)

    ### reset WITH model, creating new chunk --> second chunk [700, 1500]
    ### in third chunk time is reset to 0 ms
    ### also in this chunk do not record the first 100 ms [0, 100]
    mon.reset(model=True)
    mon.pause()
    simulate(100)

    ### start all monitors and simulate for 700 ms [100, 800]
    mon.start()
    simulate(700)

    ### Next we demonstrate getting recordings DURING SIMULATION by using
    ### get_recordings_and_clear
    ### this also resets the monitors back to their initialized state, i.e. there are no
    ### recordings and they are not started yet
    ### recordings1 consists of 3 chunks, third chunk [0, 800]
    recordings1, recording_times1 = mon.get_recordings_and_clear()

    ### Now continue simulation, creating NEW RECORDINGS, monitors are not started yet
    ### model was not reset, so time continues at 800 ms
    ### simulate for 100 ms [800, 900]
    simulate(100)

    ### start all monitors and simulate for 100 ms [900, 1000]
    mon.start()
    simulate(100)

    ### reset monitors and model, creating new chunk --> first chunk [800, 1000]
    ### simulate for 100 ms [0, 100]
    mon.reset(model=True)
    simulate(100)

    ### get recordings using get_recordings_and_clear
    ### this time directly start recording again
    ### recordings2 consists of 2 chunks, second chunk [0, 100]
    recordings2, recording_times2 = mon.get_recordings_and_clear()

    ### Now continue simulation, creating NEW RECORDINGS
    ### directly start monitors and reset model so time is reset to 0 ms
    ### simulate for 100 ms [0, 100]
    mon.start()
    mon.reset(model=True)
    simulate(100)

    ### get recordings the normal way (simultions are finished)
    ### recordings3 consists of 1 chunk [0, 100]
    recordings3 = mon.get_recordings()
    recording_times3 = mon.get_recording_times()

    ### print the idx and time lims of the recordings and the sizes of the recorded
    ### arrays
    print("#################### ALL RECORDINGS INFO ####################")
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
    print("#############################################################")

    ### plot recordings 1 consisting of 3 chunks
    for chunk in range(len(recordings1)):
        ### using plot_recordings which plots the recordings of one chunk
        PlotRecordings(
            figname=f"monitor_recordings_1_chunk{chunk}.png",
            recordings=recordings1,
            recording_times=recording_times1,
            shape=(2, 2),
            plan={
                "position": [1, 2, 3],
                "compartment": ["my_pop1", "my_pop2", "my_pop1"],
                "variable": ["v", "v", "spike"],
                "format": ["line", "line", "raster"],
            },
            chunk=chunk,
        )

    ### plot recordings 2 consisting of 2 chunks
    for chunk in range(len(recordings2)):
        ### using plot_recordings which plots the recordings of one chunk
        PlotRecordings(
            figname=f"monitor_recordings_2_chunk{chunk}.png",
            recordings=recordings2,
            recording_times=recording_times2,
            shape=(2, 2),
            plan={
                "position": [1, 2, 3],
                "compartment": ["my_pop1", "my_pop2", "my_pop1"],
                "variable": ["v", "v", "spike"],
                "format": ["line", "line", "raster"],
            },
            chunk=chunk,
        )

    ### plot recordings 3 consisting of 1 chunk
    for chunk in range(len(recordings3)):
        ### using plot_recordings which plots the recordings of one chunk
        PlotRecordings(
            figname=f"monitor_recordings_3_chunk{chunk}.png",
            recordings=recordings3,
            recording_times=recording_times3,
            shape=(2, 2),
            plan={
                "position": [1, 2, 3],
                "compartment": ["my_pop1", "my_pop2", "my_pop1"],
                "variable": ["v", "v", "spike"],
                "format": ["line", "line", "raster"],
            },
            chunk=chunk,
        )

    return 1


if __name__ == "__main__":
    main()
