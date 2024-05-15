import os 
from scipy.stats import pearsonr
eeg_raw_data_dir = 'C:/Users/stapl/Documents/CDocuments/FinalYearProject/Model/eeg_raw_data'
import numpy as np
import mne

all_channels_data = {} # Will be of length 68 as this is max channels
smallest_ch_samples = 74255 #precalculated

def gen_emp_psd(eeg_freq):   
    i = 0
    for filename in os.listdir(eeg_raw_data_dir):
        if i < 10:
            eeg_path = os.path.join(eeg_raw_data_dir, filename)
            raw = mne.io.read_raw_fif(eeg_path, preload=True)
            for ch in raw.ch_names:
                # Different samples have different time points! 
                channel_samples = raw[ch, 0:smallest_ch_samples][0] # Take first of tuple, as second item is the time
                if ch in all_channels_data.keys():
                    all_channels_data[ch].append(channel_samples)
                else:
                    all_channels_data[ch] = [channel_samples]
            i += 1

    for ch in all_channels_data.keys():
        stacked_data = np.stack(all_channels_data[ch])
        average_array = np.mean(stacked_data, axis=0)
        all_channels_data[ch] = average_array

    emp_data = np.squeeze(np.stack(list(all_channels_data.values())))
    info = mne.create_info(list(all_channels_data.keys()), sfreq=eeg_freq, ch_types='eeg')
    emp_raw = mne.io.RawArray(emp_data, info)#
    # Added concurrency, -1 will use all cores
    emp_spec = emp_raw.compute_psd(fmin=0, fmax=80, picks="all", n_jobs=-1)
    np.save('emp_spec.npy', emp_spec)
