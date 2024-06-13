import os
import numpy as np
import mne
from scipy.stats import zscore
import pandas as pd

eeg_raw_data_dir = (
    "C:/Users/stapl/Documents/CDocuments/FinalYearProject/Model/eeg_raw_data"
)

all_channels_psds = {}  # Will be of length 62 as this is max channels

smallest_ch_samples = 74255  # precalculated
observed_freq_cap = 80
n_fft = 2048

metadata = pd.read_csv("eeg_metadata.csv")

# Filter subject IDs based on the 'Study' column
schizophrenia_subjects = metadata[metadata["Study"] == "Proband with Schizophrenia"][
    "SubjectID"
].tolist()
healthy_controls = metadata[metadata["Study"] == "Healthy Control"][
    "SubjectID"
].tolist()

skipped = []


def gen_emp_psd(eeg_freq):
    for filename in os.listdir(eeg_raw_data_dir):

        subject_id = int(filename.split("_", 5)[4])

        # Add filter here, depending on which subject group you want to generate the PSD for
        # Make sure to change the name saved too
        if (subject_id) not in healthy_controls:
            continue  # Skip to next iteration

        eeg_path = os.path.join(eeg_raw_data_dir, filename)
        raw = mne.io.read_raw_fif(eeg_path, preload=True)

        # In order to average spectra, must have stationary signal, so z-score signals first
        data = raw.get_data()  # shape is (n_channels, n_samples)
        data = zscore(data, axis=1)

        # Exclude non-EEG channels
        exclude_channels = ["CB1", "CB2", "VEO", "HEO", "EKG", "EMG"]
        filtered_ch_names = [
            ch_name for ch_name in raw.ch_names if ch_name not in exclude_channels
        ]

        # Compute PSD using Welch's method for each channel
        for ch_idx, ch_name in enumerate(filtered_ch_names):

            ch_data = data[ch_idx, :]  # Get data for the specific channel

            psd, _ = mne.time_frequency.psd_array_welch(
                ch_data,
                sfreq=eeg_freq,
                fmin=0,
                fmax=80,
                n_fft=n_fft,
                n_overlap=n_fft // 2,
                window="hamming",
            )

            if ch_name in all_channels_psds:
                all_channels_psds[ch_name].append(psd)
            else:
                all_channels_psds[ch_name] = [psd]

    # Average PSDs across all subjects for each channel
    for ch_name in filtered_ch_names:
        stacked_psds = np.stack(all_channels_psds[ch_name])
        avg_psd = np.mean(stacked_psds, axis=0)
        all_channels_psds[ch_name] = avg_psd

    # Combine the averaged PSDs into a single array
    # Shape (62, freq_samples)
    emp_psd = np.stack(list(all_channels_psds.values()))

    # Save the average spectrum
    np.save("emp_spec_schiz.npy", emp_psd)
