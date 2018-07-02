%prediction script
addpath('./mesh')
addpath('./featureFunctions/nonOverlappingPolydisperseSpheres')

if ~exist('rom', 'var')
    rom = StokesROM;
end
testSamples = 101:106;

testData = StokesData(testSamples);
[~, ~, effCond, meanSqDist, ~, mll, R, R_i] = rom.predict(testData, 'local');

save('./prediction.mat', 'meanSqDist', 'mll', 'R');