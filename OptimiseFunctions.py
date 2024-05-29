import Jansen_And_Rit
from scipy import signal
import numpy as np
import BOLD_Model


def find_eeg_loss(x):
    print(x[0], x[1], x[2]) # A, B, C
    # Running cached version, so x1 x2 x3 saved to be used in optimise BOLD if same paramters input 
    x1, x2, x3, V_T_sim = run_jansen_and_rit_with_caching(x[0], x[1], x[2]) # A, B, C

    emp_spec = np.load('emp_spec.npy', allow_pickle=True)

    gen_data = V_T_sim.T
    fake_info = mne.create_info(100, sfreq=eeg_freq, ch_types='eeg')
    gen_raw = mne.io.RawArray(gen_data, fake_info)
    gen_spec = gen_raw.compute_psd(fmin=0, fmax=80, picks="all")
    # Trims the data to the same number of freq samples, marginal diff (161 to 168, as empirical over 8 seconds and generated over 2)
    freq_samples = min(gen_spec.shape[1], emp_spec.shape[1])

    # To store correlation coefficients per channel
    correlation_coefficients = []

    # Calculate per-channel correlation 
    for ch in range(gen_spec_data.shape[0]):
        gen_channel = gen_spec_data[ch]
        emp_channel = emp_spec_data[ch]
        correlation_coefficient, _ = pearsonr(gen_channel, emp_channel)
        correlation_coefficients.append(correlation_coefficient)

    # Calculate the average correlation coefficient across all channels
    correlation_coefficients = np.array(correlation_coefficients)
    average_correlation = np.mean(correlation_coefficients)

    loss = -average_correlation
    return loss

# Load the empricical (averaged) FC matrix
observed_fc_matrix = np.load("fc_matrices/averaged_fc_matrix.npy")

def find_bold_loss(x):
    print(x[0], x[1], x[2]) # A, B, C

    # If already calculated by optimise EEG, will retrieve saved
    # Otherwise, will run model again
    x1, x2, x3 = run_jansen_and_rit_with_retrieval(x[0], x[1], x[2])

    # As J&R model already run with downsampling for eeg, need to adjust downsampling rate
    adjusted_downsample = int(downsample_bold / downsample_eeg)

    # Initial conditions
    initial_conditions = np.array([[0.1, 1, 1, 1]] * num_nodes).T

    total_bold_downsampled_sims = int(total_downsampled_sims / adjusted_downsample)

    # Array to store results
    BOLD_vars = np.zeros((total_downsampled_sims + 1 , 4, num_nodes))
    BOLD_vars[0] = initial_conditions
    BOLD_temp = np.copy(initial_conditions)

    # Run simulation using Euler method, NOTE - total_downsampled_sims is the number of timepoints we have firing rates for 
    for i in range(total_downsampled_sims):
        # dt has to match the sampling frew of what you pass into the BOLD model, in this case it is the data already 
        # downsampled by eeg freq
        BOLD_temp += (1/eeg_freq) * np.array(balloon_windkessel_ode(BOLD_temp, i, x1, x2, x3))
        BOLD_vars[i] = np.copy(BOLD_temp)

    #Downsample by adjusted rate
    BOLD_vars = BOLD_vars[::adjusted_downsample]

    # Take final half of results as simulation points
    BOLD_sim_length = int(1/2 * total_bold_downsampled_sims)
    BOLD_vars_result = BOLD_vars[-BOLD_sim_length:]

    # Initialize BOLD array - will only take the final BOLD_sim_length timepoints
    BOLD_array = np.zeros((BOLD_sim_length - 1, num_nodes))

    # Generate BOLD array
    q = BOLD_vars_result[:-1, 3, :]
    v = BOLD_vars_result[:-1, 2, :]

    BOLD_array = V_0 * (k1 * (1 - q) + k2 * (1 - (q / v)) + k3 * (1 - v))

    # Pass BOLD signals through bandpass filter

    Fmin, Fmax = 0.01, 0.1
    tr = 1 / bold_freq
    a0, b0 = signal.bessel(3, [2 * tr * Fmin, 2 * tr * Fmax], btype = 'bandpass')
    BOLDfilt = signal.filtfilt(a0, b0, BOLD_array[:,:], axis = 0)

    # Calculate the FC from the filtered BOLD signal

    FC_matrix = np.corrcoef(BOLDfilt.T)

    # Calculate goodness of fit for BOLD data ####################################################

    # Flatten both and use pearson correlation to calculate goodness of fit
    pearson_corr = np.corrcoef(np.ravel(FC_matrix), np.ravel(observed_fc_matrix))[0, 1]

    loss = -pearson_corr
    return loss


