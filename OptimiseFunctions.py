import Jansen_And_Rit as JR
from scipy import signal
import numpy as np
import mne
from scipy.stats import pearsonr


def find_eeg_loss(x):
    print(
        x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10]
    )  # A, B, C, a, ad, b, r_0, r_1, r_2, alpha, beta

    x1, x2, x3, V_T_sim = JR.run_jansen_and_rit(
        x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10]
    )

    # Change this depending on what dataset optimising for
    emp_spec = np.load("emp_spec_schiz.npy", allow_pickle=True)

    gen_data = V_T_sim.T
    # Create a fake info object so mne can recognise the data as EEG signal 
    fake_info = mne.create_info(62, sfreq=JR.eeg_freq, ch_types="eeg")
    gen_raw = mne.io.RawArray(gen_data, fake_info)
    gen_spec = gen_raw.compute_psd(fmin=0, fmax=80, picks="all")

    # Trims the data to the same number of freq samples, marginal diff (161 to 168, as empirical over 8 seconds and generated over 2)
    freq_samples = min(gen_spec.shape[1], emp_spec.shape[1])
    gen_spec_data = gen_spec[:, :freq_samples]  # Shape (62, freq_samples)
    emp_spec_data = emp_spec[:, :freq_samples]  # Shape (62, freq_samples)

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


# BOLD #########################################################################################

# Load the empricical (averaged) FC matrix - change this depending on what dataset optimising for
observed_fc_matrix = np.load("fc_matrices/average_control_matrix.npy")

# BOLD MODEL ###################################################################################

import numpy as np
from sklearn import datasets

# Balloon-Windkessel Hemodynamic Model #########################################################

# Time Constants
tau_s = 0.65  # Signal decay
tau_f = 0.41  # blood inflow
tau_v = 0.98  # blood volume
tau_q = 0.98  # deoxyhemoglobin content

k = 0.32  # stiffness constant, represents resistance in veins to blood flow
E_0 = 0.4  # resting oxygen extraction rate

# Chosen Constants
k1 = 2.77
k2 = 0.2
k3 = 0.5

V_0 = 0.03  # fraction of deoxygenated blood in resting state


# ODEs for Balloon Windkessel Model
def balloon_windkessel_ode(state, t):

    # s: vasodilatory response
    # f: blood inflow
    # v: blood volume
    # q: deoxyhemoglobin content
    s, f, v, q = state

    # Constrain between 1^-10, and 100000000 so that it doesn't cause overflow on the power
    f = np.clip(f, 1e-10, 1e8)
    v = np.clip(v, 1e-10, 1e8)

    ds_dt = firing_rates[t] - s / tau_s - (f - 1) / tau_f
    df_dt = s
    dv_dt = (f - v ** (1 / k)) / tau_v
    dq_dt = ((f * (1 - (1 - E_0) ** (1 / f))) / E_0 - (q * v ** (1 / k)) / v) / tau_q

    return [ds_dt, df_dt, dv_dt, dq_dt]


# LOSS FUNCTION ################################################################################


def find_bold_loss(x):
    global firing_rates

    print(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10])

    x1, x2, x3, _ = JR.run_jansen_and_rit(
        x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10]
    )

    raw_firing_rates = x1 + x2 + x3

    # Calculate mean and standard deviation across all nodes
    mean_firing_rate = np.mean(raw_firing_rates)
    std_firing_rate = np.std(raw_firing_rates)

    # Standardize firing rates
    firing_rates = (raw_firing_rates - mean_firing_rate) / std_firing_rate

    # As J&R model already run with downsampling for eeg, need to adjust downsampling rate
    adjusted_downsample = int(JR.downsample_bold / JR.downsample_eeg)

    # Initial conditions
    initial_conditions = np.array([[0.1, 1, 1, 1]] * JR.num_nodes).T

    total_bold_downsampled_sims = int(JR.total_downsampled_sims / adjusted_downsample)

    # Array to store results
    BOLD_vars = np.zeros((JR.total_downsampled_sims + 1, 4, JR.num_nodes))
    BOLD_vars[0] = initial_conditions
    BOLD_temp = np.copy(initial_conditions)

    # Run simulation using Euler method, NOTE - total_downsampled_sims is the number of timepoints we have firing rates for
    for i in range(JR.total_downsampled_sims - 1):
        # dt has to match the sampling frew of what you pass into the BOLD model, in this case it is the data already
        # downsampled by eeg freq
        BOLD_temp += (1 / JR.eeg_freq) * np.array(balloon_windkessel_ode(BOLD_temp, i))
        BOLD_vars[i] = np.copy(BOLD_temp)

    # #Downsample by adjusted rate
    BOLD_vars = BOLD_vars[::adjusted_downsample]

    # Take final half of results as simulation points
    BOLD_sim_length = int(1 / 2 * total_bold_downsampled_sims)
    BOLD_vars_result = BOLD_vars[-BOLD_sim_length:]

    # Initialize BOLD array - will only take the final BOLD_sim_length timepoints
    BOLD_array = np.zeros((BOLD_sim_length - 1, JR.num_nodes))

    # Generate BOLD array
    q = BOLD_vars_result[:-1, 3, :]
    v = BOLD_vars_result[:-1, 2, :]

    BOLD_array = V_0 * (k1 * (1 - q) + k2 * (1 - (q / v)) + k3 * (1 - v))

    # Pass BOLD signals through bandpass filter

    Fmin, Fmax = 0.01, 0.1
    tr = 1 / JR.bold_freq
    a0, b0 = signal.bessel(3, [2 * tr * Fmin, 2 * tr * Fmax], btype="bandpass")
    BOLDfilt = signal.filtfilt(a0, b0, BOLD_array[:, :], axis=0)

    # Calculate the FC from the filtered BOLD signal

    FC_matrix = np.corrcoef(BOLDfilt.T)

    # Calculate goodness of fit for BOLD data ####################################################

    # Flatten both and use pearson correlation to calculate goodness of fit
    pearson_corr = np.corrcoef(np.ravel(FC_matrix), np.ravel(observed_fc_matrix))[0, 1]

    loss = -pearson_corr
    return loss
