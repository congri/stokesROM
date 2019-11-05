%To plot a data histogram
clear;
addpath('./mesh')
addpath('./FEM')


x = [1, 1];
nStart = 0;
nEnd = 10000;
data = StokesData(nStart:nEnd);

data.readData('xp');
modelParams = ModelParams(data.u_bc, data.p_bc);
data.interpolate(modelParams);
modelParams.fineScaleInterp(data.X_interp);
data.shiftData(true, 'p'); %shifts p to 0 at origin

%find closest interpolation point to x
diff = sum((data.X_interp{1} - x).^2, 2);
[~, i] = min(diff);

nSamples = numel(data.P);
P = zeros(0, nSamples);
for n = 1:nSamples
    P(n) = data.P{n}(i);
end

histogram(P);


