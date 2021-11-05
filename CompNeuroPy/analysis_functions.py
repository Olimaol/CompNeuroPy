import numpy as np

def hanning_split_overlap(seq, size, overlap):
    """
        splits a sequence (array) in as many hanning-windowed subsequences as possible
    
        seq: full original sequence
        size: size of subsequences to be generated from the full sequence, subsequences are hanning-windowed
        overlap: overlap of subsequences
    
        iterates over complete size of subsequences
        --> = start positions
        from startpositions iterate to reach end of fullsequence with stepsize subsequencesize-overlap
        --> for each start position n sequence positions are obtained = indizes of seq for building the subsequences
            first row:  [0, 0+(subsequencesize-overlap), 0+2*(subsequencesize-overlap), ...]
            second row: [1, 1+(subsequencesize-overlap), 1+2*(subsequencesize-overlap), ...]
        --> similar to matrix, columns = indizes for building the subsequences
    """
    #seq[i::stepsize] == seq[range(i,seq.size,stepsize)]
    return np.array([x*np.hanning(size) for x in zip(*[seq[i::size-overlap] for i in range(size)])])
    
    
def get_population_power_spectrum(spikes,presimulationTime,simulationTime,simulation_dt, spikesSamplingfrequency=250, fftSize=1024):
    """
        generates power spectrum of population spikes
        using the Welch methode (Welch,1967)
        
        spikes: ANNarchy spike dict of one population
        presimulationTime: simulation time which will not be analyzed
        simulationTime: analyzed simulation time
        
        spikesSamplingfrequency: to sample the spike times, in Hz --> max frequency = samplingfrequency / 2
        fftSize: signal size for FFT, duration (in s) = fftSize / samplingfrequency --> frequency resolution = samplingfrequency / fftSize
    """
    populationSize = len(list(spikes.keys()))
    
    if (simulationTime / 1000) < (fftSize / spikesSamplingfrequency):
        print('Simulation time has to be >=',fftSize / spikesSamplingfrequency,'s for FFT!')
        return [np.zeros(int(fftSize/2-2)),np.zeros(int(fftSize/2-2))]
    else:
        spektrum=np.zeros((populationSize,fftSize))
        for neuron in range(populationSize):
            ### sampling steps array
            spiketrain=np.zeros(int((simulationTime/1000.)*spikesSamplingfrequency))
            ### spike times as sampling steps
            idx=(((np.array(spikes[neuron])*simulation_dt)/1000.)*spikesSamplingfrequency).astype(np.int32)
            ### cut the spikes from presimulation
            idx=idx-(presimulationTime/1000.)*spikesSamplingfrequency
            idx=idx[idx>=0].astype(np.int32)
            ### sampling steps array, one if there was a spike at sampling step = spike train
            spiketrain[idx]=1
            
            ###generate multiple overlapping sequences out of the spike trains  
            spiketrainSequences=hanning_split_overlap(spiketrain,fftSize,int(fftSize/2))
        
            ###generate power spectrum
            spektrum[neuron]=np.nanmean(  np.abs(np.fft.fft(spiketrainSequences))**2  ,0)

        ###mean spectrum over all neurons
        spektrum=np.nanmean(spektrum,0)
        
        frequenzen=np.fft.fftfreq(fftSize,1./spikesSamplingfrequency)
        
        return [frequenzen[2:int(fftSize/2)],spektrum[2:int(fftSize/2)]]
        
        
def get_power_spektrum_from_time_array(arr,presimulationTime,simulationTime,simulation_dt, samplingfrequency=250, fftSize=1024):
    """
        generates power spectrum of time signal (returns [frequencies,power])
        using the Welch methode (Welch,1967)
        
        arr: time array, value for each timestep
        presimulationTime: simulation time which will not be analyzed
        simulationTime: analyzed simulation time
        
        samplingfrequency: to sample the arr, in Hz --> max frequency = samplingfrequency / 2
        fftSize: signal size for FFT, duration (in s) = fftSize / samplingfrequency --> frequency resolution = samplingfrequency / fftSize
    """
    
    if (simulationTime / 1000) < (fftSize / samplingfrequency):
        print('Simulation time has to be >=',fftSize / samplingfrequency,'s for FFT!')
        return [np.zeros(int(fftSize/2-2)),np.zeros(int(fftSize/2-2))]
    else:
        ### sampling steps array
        sampling_arr=arr[0::int((1/samplingfrequency)*1000/simulation_dt)]
        
        ###generate multiple overlapping sequences
        sampling_arr_sequences=hanning_split_overlap(sampling_arr,fftSize,int(fftSize/2))
    
        ###generate power spectrum
        spektrum=np.nanmean(  np.abs(np.fft.fft(sampling_arr_sequences))**2  ,0)
        
        frequenzen=np.fft.fftfreq(fftSize,1./samplingfrequency)
        
        return [frequenzen[2:int(fftSize/2)],spektrum[2:int(fftSize/2)]]
