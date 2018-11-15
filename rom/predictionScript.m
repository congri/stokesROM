%prediction script
addpath('./mesh')
addpath('./featureFunctions/nonOverlappingPolydisperseSpheres')
addpath('./FEM')
addpath('./rom')


if ~exist('rom', 'var')
    rom = StokesROM;
end
testSamples = 128:1012;

testData = StokesData(testSamples);
[~,~,~, meanSqDist, ~, mll, R, ~] = rom.predict(testData, 'local');

save('./prediction.mat', 'meanSqDist', 'mll', 'R');