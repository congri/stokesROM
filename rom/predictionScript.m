%prediction script
addpath('./mesh')
addpath('./featureFunctions/nonOverlappingPolydisperseSpheres')

%rom = StokesROM;
testSamples = 16:21;

testData = StokesData(testSamples);
[~, ~, ~, meanSqDist, ~, mll] = rom.predict(testData, 'local');

save('./prediction.mat', 'meanSqDist', 'mll');