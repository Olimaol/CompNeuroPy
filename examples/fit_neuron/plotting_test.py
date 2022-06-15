import numpy as np
import CompNeuroPy.analysis_functions as af


results_soll = np.load('dataRaw/generate_H_and_H_data/results.npy', allow_pickle=True).item()

recordings = results_soll['recordings']
recording_times = results_soll['data']['recording_times']

plot_list = ['1;HH_Corbit;spike;single',
             '2;HH_Corbit;v;line']
chunk = 0
time_lims = recording_times.time_lims()
idx_lims  = recording_times.idx_lims()
af.plot_recordings('1st_part.png', recordings[chunk], time_lims, idx_lims, (2,1), plot_list)


plot_list = ['1;HH_Corbit;spike;single',
             '2;HH_Corbit;v;line']
chunk = 1
time_lims = recording_times.time_lims(chunk=chunk)
idx_lims  = recording_times.idx_lims(chunk=chunk)
af.plot_recordings('2nd_part.png', recordings[chunk], time_lims, idx_lims, (2,1), plot_list)
