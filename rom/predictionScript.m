%prediction script
addpath('./mesh')
addpath('./featureFunctions/nonOverlappingPolydisperseSpheres')
addpath('./FEM')
addpath('./rom')


if ~exist('rom', 'var')
    rom = StokesROM;
end
testSamples = 0:1023;

testData = StokesData(testSamples);
[~,~,~, meanSqDist, ~, mll, R, ~] = rom.predict(testData, 'local');

save('./prediction.mat', 'meanSqDist', 'mll', 'R');