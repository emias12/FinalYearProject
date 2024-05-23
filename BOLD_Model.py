import numpy as np
from sklearn import datasets

# Balloon-Windkessel Hemodynamic Model #####################################################
# Time Constants
tau_s = 0.65 # Signal decay
tau_f = 0.41 # blood inflow
tau_v = 0.98 # blood volume
tau_q = 0.98 # deoxyhemoglobin content

k = 0.32 # stiffness constant, represents resistance in veins to blood flow
E_0 = 0.4 # resting oxygen extraction rate

# Chosen Constants 
k1 = 2.77
k2 = 0.2
k3 = 0.5

V_0 = 0.03 # fraction of deoxygenated blood in resting state 

# Calculate firing rates, V_T is EEG like response, r_0 is the sigmoid slope
#firing_rates = sigmoid(V_T, r_0) # sim_length x num_nodes

# Above is from the Neuromod, but had already got firing rates before converted to voltage above, so 
# probably makes more sense to sum x1 x2 x3 (as all need blood)
%store -r x1
%store -r x2
%store -r x3
raw_firing_rates = x1 + x2 + x3

# Calculate mean and standard deviation across all nodes
mean_firing_rate = np.mean(raw_firing_rates)
std_firing_rate = np.std(raw_firing_rates)

# Standardize firing rates
firing_rates = (raw_firing_rates - mean_firing_rate) / std_firing_rate

scaling = 1
# instead of multiplying by arbitrary factor, can standardise - divide by standard deviation, so sd becomes 1

# ODEs for Balloon Windkessel Model
def balloon_windkessel_ode(state, t):
    # s: vasodilatory response
    # f: blood inflow
    # v: blood volume
    # q: deoxyhemoglobin content
    s, f, v, q = state
    
    # NOTE - multiplying firing rate by a large constant
    ds_dt  = scaling * firing_rates[t] - s / tau_s - (f - 1) / tau_f
    df_dt = s
    dv_dt = (f - v**(1/k)) / tau_v
    dq_dt = ((f * (1 - (1 - E_0)**(1 / f))) / E_0 - (q * v**(1/k)) / v) / tau_q
    
    return [ds_dt, df_dt, dv_dt, dq_dt]