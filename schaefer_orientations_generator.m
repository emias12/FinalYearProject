% Add FieldTrip to your MATLAB path
addpath('C:/ProgramData/Microsoft/Windows/Start Menu/Programs/MATLAB R2024a/fieldtrip-20240515');

% Initialize FieldTrip defaults
ft_defaults;
%%
t = readtable('Schaefer2018_100Parcels_7Networks_order_FSLMNI152_2mm.Centroid_RAS.csv');
%%
R_column = {t.R}'; 
R_column = cell2mat(R_column); 
%%
A_column = {t.A}'; 
A_column = cell2mat(A_column); 
%%
S_column = {t.S}';
S_column = cell2mat(S_column); 
%%
surfnorm([R_column, A_column, S_column])
%%
R_column(1,1) % (rows, cols). Indexed from 1!
%%
[x1, x2, x3] = surfnorm([R_column, A_column, S_column])
%%
R_normalised = zeros(100, 1);
A_normalised = zeros(100, 1);
S_normalised = zeros(100, 1);

for i = 1:100
    coefficient_sums = [sum(x1(i, :)), sum(x2(i, :)), sum(x3(i, :))]

    R = sum(coefficient_sums(:, 1));
    A = sum(coefficient_sums(:, 2));
    S = sum(coefficient_sums(:, 3));

    norm_factor = sqrt(R^2 + A^2 + S^2);
    
    R_normalised(i) = R / norm_factor;
    A_normalised(i) = A / norm_factor;
    S_normalised(i) = S / norm_factor;  
end
%%
% Combine into 1 vector 
combined_coefficients = [R_normalised, A_normalised, S_normalised];