import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
import mne 
from sklearn import datasets
from scipy.stats import pearsonr
import hashlib
import os
import gc
import scipy.stats, scipy.io


# Parameters ###################################################################

max_firing_rate = 5         # (per second)
mean_firing_threshold = 2.5 # [Theta] (mV), half of the maximum response of the pop

# Sigmoid slopes (mV^-1) 
r_0 = 0.56
r_1 = 0.56
r_2 = 0.56

# Inverse time constants (s^-1)
# Smaller value, slower effect.. 
a = 100 # excitatory 
ad = 50 # long-range excitatory
b = 50  # inhibitory

# Maximum amplitudes of post-synaptic potential (PSPs) (mV)
A = 3.25 # excitatory
B = 22.0 # inhibitory

# Connectivity constants
C = 135         # Global synaptic connectivity
C1 = C          # Connectivity between Pyramidal and Excitatory
C2 = 0.8 * C    # Connectivity between Excitatory and Pyramidal
C3 = 0.25 * C   # Connectivity between Pyramidal and Inhibitory
C4 = 0.25 * C   # Connectivity between Inhibitory and Pyramidal

# Modified Jansen & Rit Parameters 
# Cholinergic modulation of inhibitory circuits and the segregation/integration balance

# Both as multiples of C
alpha = 0 # excitatory gain, connectivity between long-range pyramidal 
beta = 0  # inhibitory gain, connectibity between inhibitory and excitatory interneuron (short range)

# Structural Connectivity Matrix
num_nodes = 100
SC = np.genfromtxt('SC_in_Schaefer-100.csv', delimiter=',')


# Sigmoid Function - transforms the postynaptic potential (PSP) into an average pulse density 
# v is the average psp, r is the slope of the sigmoid function
def sigmoid(v, r):
    return max_firing_rate / (1 + np.exp(r * (mean_firing_threshold - v))) # output is num_nodes x 1

# x3 is a vector of size num_nodes by 1
def calculate_zi(x3):
    return np.dot(SC, x3) # output num_nodes x 1

# System of Equations ########################################################  

# Expanded Jansen & Rit Model
def system_of_equations(x):
    x0, y0, x1, y1, x2, y2, x3, y3 = x

    # Noise  - uncorrelated Gaussian-distributed noise with mean 2 and standard deviation 2
    noise = np.random.normal(2,2,num_nodes) # produces a num_nodes x 1 vector 

    dx0dt = y0
    dy0dt = A * a * (sigmoid(C2 * x1 - C4 * x2 + C * alpha * calculate_zi(x3), r_0)) - 2 * a * y0 - a**2 * x0
    dx1dt = y1
    dy1dt = A * a * (noise + sigmoid(C1 * x0 - C * beta * x2, r_1)) - 2 * a * y1 - a**2 * x1
    dx2dt = y2
    dy2dt = B * b * sigmoid(C3 * x0, r_2) - 2 * b * y2 - b**2 * x2
    dx3dt = y3
    dy3dt = A * ad * (sigmoid(C2 * x1 - C4 * x2 + C * alpha * calculate_zi(x3), r_0)) - 2 * ad * y3 - ad**2 * x3
    return [dx0dt, dy0dt, dx1dt, dy1dt, dx2dt, dy2dt, dx3dt, dy3dt] # num_nodes x 8 matrix output 

eeg_freq = 1000
bold_freq = 0.5

# Simulation parameters
dt = 0.001 # Step size
transient = 60  # Simulation duration for stabilizing (with Euler method)
sim_length = 600 # Simulation time points (to plot)

downsample_eeg = (1 / dt) / eeg_freq
downsample_bold = (1 / dt) / bold_freq

# Initial conditions
initial_conditions = np.ones((8, num_nodes)) * 0.5

total_sims = int((transient + sim_length/ dt) - 1)
total_downsampled_sims = int(total_sims / downsample_eeg)

# Leadfield ###################################################################

# Load leadfield matrix

leadfield = scipy.io.loadmat('reshaped_leadfield.mat')
leadfield = leadfield['leadfield'] # Shape (100, 62, 3)
# leadfield = np.linalg.norm(leadfield, axis=-1).T
leadfield = np.sum(leadfield, axis=-1).T

# Normalise leadfield

# leadfield_sum = np.sum(leadfield, axis=1)  # Sum along the second axis (summing each row)
# leadfield = leadfield / leadfield_sum[:, np.newaxis]  # Normalise each row

nb_sources = 100
nb_sensors = 62

def pass_through_leadfield(sim_data):
    sim_eeg_sources = sim_data.T
    sim_eeg_sensors = leadfield @ sim_eeg_sources
    # sim_eeg_zscores = scipy.stats.zscore(sim_eeg_sensors, axis=0)
    return sim_eeg_sensors.T

# Memoization ###################################################################

cache_dir = "cache"

def generate_cache_key(params):
    rounded_params = []
    for param in params:
        if isinstance(param, float):
            rounded_params.append(f"{param:.1f}")
        else:
            rounded_params.append(str(param))
    params_str = "_".join(rounded_params)
    return hashlib.md5(params_str.encode()).hexdigest()

def cache_result(params, result):
    key = generate_cache_key(params)
    file_path = os.path.join(cache_dir, key + ".npy")
    np.save(file_path, result)
    gc.collect()

def load_cached_result(params):
    key = generate_cache_key(params)
    file_path = os.path.join(cache_dir, key + ".npy")
    if os.path.exists(file_path):
        return np.load(file_path, allow_pickle=True)
    return None


# Run Jansen & Rit Model ########################################################

# Original A = 3.25 # excitatory, B = 22.0 # inhibitory, C = 135
# B > A. b < a.
# b = 50, ad = 50, a = 100
# r_0, r_1, r_2 = 0.56
# mean_firing_threshold = 6
# max_firing_rate = 5

 # A, B, C, a, ad, b, r_0, r_1, r_2, alpha, beta

def run_jansen_and_rit(A_inp=A, B_inp=B, C_inp=C, a_inp=a, ad_inp=ad, b_inp=b, r_0_inp=r_0, r_1_inp=r_1, r_2_inp=r_2, alpha_inp=alpha, beta_inp=beta):
    global A
    global B
    global C
    global a
    global ad
    global b
    global r_0
    global r_1
    global r_2
    global alpha
    global beta

    A = A_inp
    B = B_inp
    C = C_inp
    a = a_inp
    ad = ad_inp
    b = b_inp
    r_0 = r_0_inp
    r_1 = r_1_inp
    r_2 = r_2_inp
    alpha = alpha_inp
    beta = beta_inp

    # Array to store results
    sol = np.zeros((total_downsampled_sims, 8, num_nodes))
    sol[0,:,:] = np.copy(initial_conditions) #First set of initial conditions
    y_temp = np.copy(initial_conditions)

    # Run simulation using Euler method
    for i in range(1, total_sims):
        y_temp += dt * np.array(system_of_equations(y_temp))
        if i % downsample_eeg == 0:
            sol[int(i/downsample_eeg) - 1] = np.copy(y_temp)

    x1 = sol[:-1, 2]
    x2 = sol[:-1, 4]
    x3 = np.apply_along_axis(calculate_zi, axis=1, arr=sol[:-1, 6])

    # eeg_freq is 1000Hz, i.e. 1000 points per second. So 2000 points in 2 seconds. 
    time_points_in_2_secs = int(2 * eeg_freq)

    V_T_sim = (C2 * x1[-time_points_in_2_secs:] - C4 * x2[-time_points_in_2_secs:]
                                      + C * alpha * x3[-time_points_in_2_secs:])

    # # With vectorised operations, Calculate V_T_sim directly for the desired time points
    # V_T_sim = pass_through_leadfield(C2 * x1[-time_points_in_2_secs:] - C4 * x2[-time_points_in_2_secs:]
    #                                   + C * alpha * x3[-time_points_in_2_secs:])

    return(x1, x2, x3, V_T_sim)


def run_jansen_and_rit_with_retrieval(A_inp, B_inp, C_inp, a_inp, ad_inp, b_inp, r_0_inp, r_1_inp, r_2_inp, alpha_inp, beta_inp):
    params = [A_inp, B_inp, C_inp, a_inp, ad_inp, b_inp, r_0_inp, r_1_inp, r_2_inp, alpha_inp, beta_inp]

    cached_result = load_cached_result(params)
    if cached_result is not None:
        print("cached")
        return cached_result
    else:
        print("not cached")
        x1, x2, x3, _  = run_jansen_and_rit(A_inp, B_inp, C_inp, a_inp, ad_inp, b_inp, r_0_inp, r_1_inp, r_2_inp, alpha_inp, beta_inp)
        return (x1, x2, x3)


def run_jansen_and_rit_with_caching(A_inp, B_inp, C_inp, a_inp, ad_inp, b_inp, r_0_inp, r_1_inp, r_2_inp, alpha_inp, beta_inp):
    x1, x2, x3, V_T_sim = run_jansen_and_rit(A_inp, B_inp, C_inp, a_inp, ad_inp, b_inp, r_0_inp, r_1_inp, r_2_inp, alpha_inp, beta_inp)
    # cache_result([A_inp, B_inp, C_inp, a_inp, ad_inp, b_inp, r_0_inp, r_1_inp, r_2_inp, alpha_inp, beta_inp], [x1, x2, x3])
    return (x1, x2, x3, pass_through_leadfield(V_T_sim))
