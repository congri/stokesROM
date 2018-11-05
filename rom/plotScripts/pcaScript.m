% Script to perform pca analysis on Stokes pressure response to see effective
% output dimension
clear;
addpath('./mesh')
addpath('./FEM')

data = StokesData(0:1012);
data.readData('xp');

params = ModelParams(data.u_bc, data.p_bc);
params.fineGridX = (1/256)*ones(1, 256);
params.fineGridY = params.fineGridX;


data.interpolate(params);
P = cell2mat(data.P);
clear data;


[coeff, score, latent, tsquared, explained, mu] = pca(P');

figure
semilogy(cumsum(explained))
axis tight
