"""
This example demonstrates how to use the DBSstimulator class to implement DBS in a
network. It is shown how to create a DBSstimulator, how to use it and how to update
pointers. In this simple example only the depolarization of the stimulated population is
demostrated. All other possible DBS mechanisms are demonstrated in the other example
"dbs_stimulator.py".
"""
from ANNarchy import Population, Izhikevich, compile, simulate
from CompNeuroPy import DBSstimulator

from ANNarchy import setup
from CompNeuroPy import CompNeuroMonitors, PlotRecordings

setup(dt=0.1)

# create populations
population1 = Population(10, neuron=Izhikevich, name="my_pop1")
population2 = Population(10, neuron=Izhikevich, name="my_pop2")

# create DBS stimulator
dbs = DBSstimulator(
    stimulated_population=population1,
    population_proportion=0.5,
    dbs_depolarization=30,
    auto_implement=True,
)

# if you work with names of populations/projections everything will work, but if you
# want to work with pointers you have to update them after calling the DBSstimulator
population1, population2 = dbs.update_pointers(pointer_list=[population1, population2])

# compile network
compile()

# create monitors
monitors = CompNeuroMonitors({"my_pop1": "v", "my_pop2": "v"})
monitors.start()

# run simulation
# 1000 ms without dbs
simulate(1000)
# 1000 ms with dbs
dbs.on()
simulate(1000)
# 1000 ms without dbs
dbs.off()
simulate(1000)

# plot recordings
PlotRecordings(
    figname="dbs_stimulator_simple.png",
    recordings=monitors.get_recordings(),
    recording_times=monitors.get_recording_times(),
    chunk=0,
    shape=(2, 1),
    plan={
        "position": [1, 2],
        "compartment": ["my_pop1", "my_pop2"],
        "variable": ["v", "v"],
        "format": ["matrix", "matrix"],
    },
)
