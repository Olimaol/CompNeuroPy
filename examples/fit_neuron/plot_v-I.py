import pylab as plt
import numpy as np
from CompNeuroPy import create_dir

### load data
### from generate_H_and_H_data
results_target = np.load('dataRaw/generate_H_and_H_data/results.npy', allow_pickle=True).item()
sp_target = results_target['sim'][0]
recordings_target = results_target['recordings']
recording_times_target = results_target['data']['recording_times']


### from optimization
sim_id=1
best = np.load('dataRaw/parameter_fit/best_'+str(sim_id)+'.npy', allow_pickle=True).item()
results_opt = best['results']
sp_opt = results_opt['sim'][0]
recordings_opt = results_opt['recordings']
recording_times_opt = results_opt['data']['recording_times']


### combine data
times, data_arr_target = recording_times_target.combine_chunks(recordings_target, sp_target.kwargs[0]['pop']+';v', mode='consecutive')
_    , data_arr_opt    = recording_times_opt.combine_chunks(recordings_opt, sp_opt.kwargs[0]['pop']+';v', mode='consecutive')


### obtain simulation times and stimulation current from simulation protocol
time_lims_0 = recording_times_target.time_lims(chunk=0)
time_lims_1 = recording_times_target.time_lims(chunk=1)
simulation_times = np.arange(time_lims_0[0], np.diff(time_lims_0)+np.diff(time_lims_1), recordings_target[0]['dt'])# both chunks combined
stimulation_current = simulation_times.copy()
for i in range(3):
    #time_before=sum([sp_target[j]['t1']+sp_target[j]['t2'] for j in range(i)])
    #stimulation_current[simulation_times>time_before]=sp_target[i]['a1']
    #time_before+=sp_target[i]['t1']
    #stimulation_current[simulation_times>time_before]=sp_target[i]['a2']
    

    time_before=sum([sp_target.kwargs[j]['t1']+sp_target.kwargs[j]['t2'] for j in range(i)])
    stimulation_current[simulation_times>time_before]=sp_target.kwargs[i]['a1']
    time_before+=sp_target.kwargs[i]['t1']
    stimulation_current[simulation_times>time_before]=sp_target.kwargs[i]['a2']

### plot results
plt.figure(dpi=300)
plt.subplot(311)
y_target = np.clip(data_arr_target, None, 0)
plt.plot(times, y_target, color='k', label='target', lw=0.3)
plt.ylim(min([data_arr_target.min(),data_arr_opt.min()]),0)
plt.ylabel('v [mV]')
plt.legend()
plt.subplot(312)
y_opt = np.clip(data_arr_opt, None, 0)
plt.plot(times, y_opt, color='grey', label='ist', lw=0.3)
plt.ylim(min([data_arr_target.min(),data_arr_opt.min()]),0)
plt.ylabel('v [mV]')
plt.legend()
plt.subplot(313)
plt.plot(simulation_times, stimulation_current)
plt.ylabel('$I_{app}$ [pA]')
plt.xlabel('time [ms]')

### save
create_dir('results/parameter_fit')
plt.savefig('results/parameter_fit/plot_v_I.svg')


