%prediction script
addpath('./mesh')
addpath('./featureFunctions/nonOverlappingPolydisperseSpheres')
addpath('./FEM')
addpath('./rom')
addpath('./comp')


if ~exist('rom', 'var')
    rom = StokesROM;
end
testSamples = 0:5;

testData = StokesData(testSamples);
[m,v,~, meanSqDist, ~, mll, R, ~] = rom.predict(testData, 'local');

save('./prediction.mat', 'meanSqDist', 'mll', 'R');