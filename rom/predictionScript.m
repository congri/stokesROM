%prediction script
addpath('./mesh')
addpath('./featureFunctions/nonOverlappingPolydisperseSpheres')
addpath('./FEM')
addpath('./rom')


if ~exist('rom', 'var')
    rom = StokesROM;
end
testSamples = 0:128;

testData = StokesData(testSamples);
[~,~,~, meanSqDist, ~, mll, R, ~] = rom.predict(testData, 'local');

save('./prediction.mat', 'meanSqDist', 'mll', 'R');