%prediction script
addpath('./mesh')
addpath('./featureFunctions/nonOverlappingPolydisperseSpheres')
addpath('./FEM')
addpath('./rom')


if ~exist('rom', 'var')
    rom = StokesROM;
end
testSamples = 16:115;

testData = StokesData(testSamples);
[~,~,~, meanSqDist, ~, mll, R, ~] = rom.predict(testData, 'local');

save('./prediction.mat', 'meanSqDist', 'mll', 'R');