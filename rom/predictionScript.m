%prediction script
addpath('./mesh')
addpath('./featureFunctions/nonOverlappingPolydisperseSpheres')

%Boundary condition fields
p_bc = @(x) 0.0;
%influx?
u_bc{1} = 'u_x=0.0-2.0*x[1]';
u_bc{2} = 'u_y=1.0-2.0*x[0]';

%rom = StokesROM;
testSamples = 0:15;

testData = StokesData(testSamples);
[~, ~, ~, meanSqDist, ~, mll] = rom.predict(testData, 'local');

save('./prediction.mat', 'meanSqDist', 'mll');