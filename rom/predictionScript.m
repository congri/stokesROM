%prediction script
addpath('./mesh')
addpath('./featureFunctions/nonOverlappingPolydisperseSpheres')

if ~exist('rom', 'var')
    rom = StokesROM;
end
testSamples = 128:256;

testData = StokesData(testSamples);
[~,~,~, meanSqDist, ~, mll, R, ~] = rom.predict(testData, 'local');

save('./prediction.mat', 'meanSqDist', 'mll', 'R');