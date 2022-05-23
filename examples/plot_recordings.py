import numpy as np
from CompNeuroPy import plot_recordings
import pylab as plt


recordings        = np.load('dataRaw/run_and_monitor_simulations/recordings.npy', allow_pickle=True)
recording_timings = np.load('dataRaw/run_and_monitor_simulations/recording_times.npy', allow_pickle=True).item()


def plot_chunk(chunk):
    ### with plot_recodings one can easily plot the recodings of one chunk
    ### plot_recordings needs the time limits (ins ms) and the idx limits for the data to plot, which can be obtained for example with the recording_timings object (they can of course also be set manually to specific values)
    time_lims = recording_timings.time_lims(chunk=chunk)
    idx_lims  = recording_timings.idx_lims(chunk=chunk)
    ### the last two arguments of plot_recordings define how subplots are arranged and which recordings are shown in which subplot (plot_list)
    structure = (2,2)
    ### the plot_list entries consist of strings with following format:
    ### 'sub_plot_nr;model_compartment;recorded_data;plotting_specifications'
    plot_list = ['1;first_poisson;spike;hybrid',
                 '2;second_poisson;spike;hybrid',
                 '3;first_poisson;p;line',
                 '4;second_poisson;p;line']
                 
    plot_recordings('results/my_two_poissons_chunk_'+str(chunk)+'_period_all.png', recordings[chunk], time_lims, idx_lims, structure, plot_list)
    
    
def plot_period(chunk,period):
    time_lims = recording_timings.time_lims(chunk=chunk, period=period)
    idx_lims  = recording_timings.idx_lims(chunk=chunk, period=period)
    structure = (2,2)
    plot_list = ['1;first_poisson;spike;hybrid',
                 '2;second_poisson;spike;hybrid',
                 '3;first_poisson;p;line',
                 '4;second_poisson;p;line']
                 
    plot_recordings('results/my_two_poissons_chunk_'+str(chunk)+'_period_'+str(period)+'.png', recordings[chunk], time_lims, idx_lims, structure, plot_list)
                 


for chunk in range(len(recordings)):
    try:
        plot_chunk(chunk)
    except:
        ### chunks with pauses in recordings at the beginning or at the end are fine, but pauses in the middle can not be plotted (no data available for the middle of the plot)
        ### one can specify the periods which should be plotted
        for period in [0,1]:
            plot_period(chunk,period)


