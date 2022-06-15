from CompNeuroPy import opt_neuron
from CompNeuroPy.system_functions import load_object
from CompNeuroPy.neuron_models import Izhikevich2007
from hyperopt import hp
import numpy as np

### load the experiment and results (target data)
results_soll = np.load('dataRaw/generate_H_and_H_data/results.npy', allow_pickle=True).item()
experiment = load_object('dataRaw/generate_H_and_H_data/experiment')


### define how the loss should be calculated
def get_loss(results_ist, results_soll):
    """
        results: results dictionary which contains recordings dictionary and simulation paradigm of experiment
            has to contain spike recordings (of one neuron)
        
        transforms spike recordings of ist and soll data into interspike intervals (unequal number of spikes --> fill up with zeros)
    """
    
    ### The get_loss calculations require following results (i.e. experiment)
    ### results contain recordings with two chunks
    assert len(results_soll['recordings'])==2 and len(results_ist['recordings'])==2, 'ERROR get_loss: Wrong number of recording chunks (2 required)!'
    ### results contain simulation protocols of three simulations, which are current step simulations
    ### results contain current_step simulation which was run 3 times
    assert results_soll['sim'][0].name=='current_step' and results_ist['sim'][0].name=='current_step', 'ERROR get_loss: Experiment has to consist of a current_step simulation run 3 times!'
    assert len(results_soll['sim'][0].start)==3 and len(results_ist['sim'][0].start)==3, 'ERROR get_loss: Experiment has to consist of a current_step simulation run 3 times!'
 

    ### get required data
    rec={}
    sp={}
    rec['ist']  = results_ist['recordings']
    rec['soll'] = results_soll['recordings']
    sp['ist']  = results_ist['sim'][0]
    sp['soll'] = results_soll['sim'][0]
    
        
    ### get ISIsof the first simulation
    ### first run of the simulation is in first chunk of recordings (before reset), spike times of first (the only) neuron
    run=0; chunk=0; neuron=0
    spike_times={}
    isi_arr={}
    for cond in ['ist', 'soll']:
        ### get spike time steps from recordings
        pop_name = sp[cond].kwargs[run]['pop']
        spike_times[cond] = np.array(rec[cond][chunk][pop_name+';spike'][neuron])
        
        ### only use spike times from second half of current_pulse simulation (after t1), start time in time steps
        start_time = int(sp[cond].kwargs[run]['t1']/rec[cond][chunk]['dt'])
        spike_times[cond] = spike_times[cond][spike_times[cond]>start_time]
        
        ### calculate inter spike intervals
        if len(spike_times[cond])>0:
            isi_arr[cond]  = np.diff(np.concatenate([np.zeros(1),np.array(spike_times[cond])]))
        else:
            isi_arr[cond] = np.zeros(1)
    
    ### check number of spikes
    nr_diff = len(isi_arr['ist'])-len(isi_arr['soll'])
    
    ### isi loss
    if np.absolute(nr_diff)>0:
        isi_loss = np.absolute(nr_diff)**2
    else:
        z_score_sd = np.std(np.concatenate([isi_arr['soll'],isi_arr['ist']]))+1e-20
        isi_loss = np.sqrt(np.mean(((isi_arr['soll']-isi_arr['ist'])/z_score_sd)**2))
        
        
    ### membrane potential from third run of the simulation
    run=2; chunk=1; neuron=0
    
    membrane_potential = {}
    for cond in ['ist', 'soll']:
        pop_name = sp[cond].kwargs[run]['pop']
        membrane_potential[cond] = rec[cond][chunk][pop_name+';v']
        
        ### only take time intervall of t1
        start_time = sp[cond].start[run]/rec[cond][chunk]['dt']
        end_time = start_time + sp[cond].kwargs[run]['t1']/rec[cond][chunk]['dt']
        membrane_potential[cond] = membrane_potential[cond][int(start_time):int(end_time)]
        
    ### membrane loss
    z_score_sd = np.std(np.concatenate([membrane_potential['soll'],membrane_potential['ist']]))+1e-20
    membrane_loss = np.sqrt(np.mean(((membrane_potential['soll']-membrane_potential['ist'])/z_score_sd)**2))
    
    
    return [isi_loss, membrane_loss]
    
    
### define the variable space
fitting_variables_space = [
                           hp.uniform('a', 0, 0.5),
                           hp.uniform('b', -1, 1),
                           hp.uniform('c', -70, -45),
                           hp.uniform('d', 0, 100),
                           hp.uniform('k', 0, 0.5),
                           hp.uniform('v_t', -55, -50)
                          ]
### constant parameters and initial values
const_params = {
                'v_peak': 30,
                'v_r': -69.8,
                'C': 1,
                'I_app': 1,
                'v': -69.8,
                'u': 0
               }
               
               
### run optimization
sim_id = 1
opt = opt_neuron(results_soll, experiment, get_loss, fitting_variables_space, const_params, compile_folder_name='annarchy_raw_Izhikevich_'+str(sim_id), num_rep_loss=1, neuron_model=Izhikevich2007)
opt.run(max_evals=200, results_file_name='best_'+str(sim_id)+'.npy')
print('\nresults:\n')
for key,val in opt.results.items():
    print(key, val, '\n')

