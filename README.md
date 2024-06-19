# Final Year Project
A Multimodal Whole Brain Model on Altered Brain Signals in Schizophrenia 

Schizophrenia is a debilitating and complex neuropsychiatric disorder with limited treatment
effectiveness. Whole-brain modelling enables the simulation of brain data through mathematical
models. Changing parameters of the model and noticing the effects can help bridge the gap between observed neuroimaging alterations and synaptic-level differences arising from variances in
neurotransmitters, cell structure, and gene expression, which is crucial for targeted medication
development. We employ the Jansen & Rit whole-brain model, focusing on neural populations
of pyramidal neurons and excitatory and inhibitory interneurons as local cortical circuits to investigate the differences in postsynaptic potential (PSP) amplitude, duration, connectivity between populations, long-range excitatory gain and local inhibitory feedback between control and
schizophrenia groups. Our model generates both electroencephalogram (EEG) data, providing
temporal resolution, and functional magnetic resonance imaging (fMRI) blood oxygenation level
dependent (BOLD) data, offering superior spatial resolution, to provide a more comprehensive understanding of underlying neural dynamics. Given the model complexity and indifferentiability, we
investigate different Bayesian optimisation techniques including Gaussian processes and gradient
boosted regression trees to find optimal parameter combinations. Our model reveals a decrease in
inhibitory PSP amplitudes and durations, and reduced local inhibitory feedback in schizophrenia
groups. This is supported by empirical findings of reduced GABA levels, GABA-related genes, and
deficient EEG gamma band oscillations in schizophrenic brains. While BOLD data showed a significant disparity in connectivity favouring control groups, EEG data revealed the opposite trend.
Given the temporal and spatial differences between these neuroimaging modalities, we hypothesise
a decrease in global connectivity and feedback from sensory nodes, and an increase in local connectivity within association nodes in schizophrenia. These findings support theories of dysconnectivity
and excitatory-inhibitory imbalance, offering insights that could inform the development of more
effective treatments

## Navigating the repository

### Jansen_And_Rit.py
- Contains main simulation code including parameter initialisation, system of equations, Euler method simulation, leadfield application

### OptimiseFunctions.py
- Contains the loss functions for EEG and BOLD
- Also includes Balloon-Windkessel hemodynamic model

### leadfield_generator.m
- Matlab file that generated the leadfield matrix

### CollectData.ipynb
- Contains code used to extract the BOLD NIFTI files using boto3 aws
- Also contains code to retrieve relevant files from EEG data

### FCMatrices.ipynb
- All code relating to computing functional connectivity matrices
- This includes using NiftLabelsMasker to compute per-subject FC matrices and averaged FC matrix
- Also contains code to filter fc_matrices from artefact, or subject group

### Generate_Empirical_PSD.py
- Contains function to compute PSD, averaged per-channel across subjects
- Can adjust to vary subject group

### Generate_orientations.ipynb
- Contains the code to assign orientations to each centroid
- Also contains code to generate plot of normals for each centroid as seen in the report

### OptimiseAll.ipynb
- This just gives an idea of how the Bayesian optimisation is run
- Many more configuratins were experimented but not included
- To plot convergence or partial dependency plots use plot_convergence(res_output) or plot_objective(res_output)

### Plot_EEG.ipynb
- Has 3 plots
- First is all eeg channels in one plot, can filter to desired channels
- Second is all channels individually
- Third is an interactive plot with mne - should pop out into separate window

### plotbrain.ipynb
- Plots centroid locations on glass brain, used for diagram in report

### psdplot.ipynb
- This is just some playing around code for plotting the PSDs for empirical and simulated data 

## Files

### reshaped_leadfield.mat
- Resulting leadfield matrix, reshaped to be compatible in the code

### emp_spec_control.npy & emp_spec_schiz.npy
- The averaged empirical PSDs for both the control group and schizophrenia group
- Loss was computed in relation to these

### orientations.npy
- Each centroids orientation as generated in Generate_orientations.ipynb

### SC_in_Schaefer-100.csv
- Schaefer 100 parcellation

### Schaefer2018_100Parcels_7Networks_order_FSLMNI152_2mm.Centroid_RAS.csv
- Contains centroid labels and coordinates
