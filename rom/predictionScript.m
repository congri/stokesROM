%prediction script
addpath('./mesh')
addpath('./featureFunctions/nonOverlappingPolydisperseSpheres')
addpath('./FEM')
addpath('./rom')
addpath('./comp')


if ~exist('rom', 'var')
    rom = StokesROM;
end
testSamples = 16:32;

testData = StokesData(testSamples);
[~,predVar,~, meanSqDist, ~, mll, R, ~] = rom.predict(testData, 'local');

save('./prediction.mat', 'meanSqDist', 'mll', 'R');