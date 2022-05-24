from ANNarchy import Population, Izhikevich, setup, simulate, get_population
from CompNeuroPy import generate_model, Monitors
import pylab as plt
import numpy as np

setup(dt=0.1)

def create_model():
    a=Population(1, neuron=Izhikevich(), name='my_pop1')
    a.i_offset=10
    b=Population(1, neuron=Izhikevich(), name='my_pop2')


my_model = generate_model(model_creation_function=create_model,  
                          name='my_model',                       
                          description='my simple example model',
                          compile_folder_name='annarchy_my_model')


monitor_dictionary = {'pop;my_pop1':['v', 'spike'], 'pop;my_pop2':['v']}
mon = Monitors(monitor_dictionary)

### first chunk, one period
simulate(100) # 100 ms not recorded
mon.start()   # start all monitors
simulate(100) # 100 ms recorded

### second chunk, two periods
mon.reset()   # model reset, beginning of new chunk
simulate(100) # 100 ms recorded (monitors were active before reset --> still active)
mon.pause(['pop;my_pop1'])   # pause recordings of my_pop1
simulate(100) # 100 ms not recorded
mon.start()   # start all monitors
simulate(50)  # 50 ms recorded
get_population('my_pop1').i_offset=50 # increase activity during last period
get_population('my_pop2').i_offset=50 # increase activity during last period
simulate(50)  # 50 ms recorded

recordings = mon.get_recordings()
y1 = recordings[0]['my_pop1;v'] ### variable v of my_pop1 from 1st chunk
y2 = recordings[1]['my_pop1;v'] ### variable v of my_pop1 from 2nd chunk


recording_times = mon.get_recording_times()
### Let's get the recoding times of my_pop1 (had a pause in the second chunk, in contrast to my_pop2)
### for the 1st chunk no period has to be defined, because there is only one
print('\nrecording time limits')
print('{:<25} {:<18} {:<18}'.format('', 'in ms','as index (dt=0.1)'))
### start and end of 1st chunk in ms and as indizes
print('{:<25} {:<18} {:<18}'.format('1st chunk',str(recording_times.time_lims(chunk=0, compartment='my_pop1')),str(recording_times.idx_lims(chunk=0, compartment='my_pop1'))))
### start and end of 2nd period of 2nd chunk in ms and as indizes
print('{:<25} {:<18} {:<18}'.format('2nd chunk, 2nd period',str(recording_times.time_lims(chunk=1, period=1, compartment='my_pop1')),str(recording_times.idx_lims(chunk=1, period=1, compartment='my_pop1'))))


### The indizes from recording_times can be used for the arrays from get_recordings
start_time = recording_times.time_lims(chunk=0, compartment='my_pop1')[0]
start_idx  = recording_times.idx_lims(chunk=0, compartment='my_pop1')[0]
end_time   = recording_times.time_lims(chunk=0, compartment='my_pop1')[1]
end_idx    = recording_times.idx_lims(chunk=0, compartment='my_pop1')[1]
x1 = np.arange(start_time, end_time, recordings[0]['dt'])
y1 = y1[start_idx:end_idx]

start_time = recording_times.time_lims(chunk=1, period=1, compartment='my_pop1')[0]
start_idx  = recording_times.idx_lims(chunk=1, period=1, compartment='my_pop1')[0]
end_time   = recording_times.time_lims(chunk=1, period=1, compartment='my_pop1')[1]
end_idx    = recording_times.idx_lims(chunk=1, period=1, compartment='my_pop1')[1]
x2 = np.arange(start_time, end_time, recordings[0]['dt'])
y2 = y2[start_idx:end_idx]

plt.figure()
plt.subplot(211)
plt.plot(x1,y1)
plt.subplot(212)
plt.plot(x2,y2)
plt.savefig('monitor_recordings.svg')


### if there are no pauses in between recordings one can combine recordings of multiple chunks, here for example for my_pop2
times, data_arr = recording_times.combine_chunks(recordings, 'my_pop2;v', mode='consecutive')

plt.figure()
plt.plot(times,data_arr)
plt.savefig('monitor_recordings2.svg')


### console output of this file:

