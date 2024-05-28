% Add FieldTrip to your MATLAB path
addpath('C:/ProgramData/Microsoft/Windows/Start Menu/Programs/MATLAB R2024a/fieldtrip-20240515');

% Initialize FieldTrip defaults
ft_defaults;
%%
headmodel = ft_read_headmodel('standard_bem.mat')
ft_plot_headmodel(headmodel, 'facealpha', 0.3)
hold on
elec = ft_read_sens('eeg_data.elc'); 
ft_plot_sens(elec)
t = readtable('Schaefer2018_100Parcels_7Networks_order_FSLMNI152_2mm.Centroid_RAS.csv');
pos = t{:, ["R", "A", "S"]};

%%
ft_plot_cloud(pos, [], 'radius', 10, 'scalerad', 'no');
%%
cfg = [];
cfg.sourcemodel = [];
cfg.pos = pos;
cfg.headmodel = headmodel;
cfg.elec = elec;
cfg.channel = {'Fp1','Fpz','Fp2','AF3','AF4','F7','F5','F3','F1','FZ','F2','F4','F6','F8','FT7',...
    'FC5','FC3','FC1','FCz','FC2','FC4','FC6','FT8','T7','C5','C3','C1','Cz','C2','C4','C6','T8',...
    'M1','TP7', 'TP8', 'CP5','CP3','CP1','CPz','CP2','CP4','CP6','M2','P7','P5','P3','P1','PZ','P2','P4',...
    'P6','P8','PO7','PO5','PO3','POz','PO4','PO6','PO8','O1','Oz','O2'}
leadfield = ft_prepare_leadfield(cfg);

disp(leadfield);