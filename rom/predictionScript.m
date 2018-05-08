%prediction script
addpath('./mesh')
addpath('./featureFunctions/nonOverlappingPolydisperseSpheres')

%Boundary condition fields
p_bc = @(x) 0;
%influx?
u_bc{1} = 'u_x=-0.8 + 2.0*x[1]';
u_bc{2} = 'u_y=-1.2 + 2.0*x[0]';

%rom = StokesROM;
testSamples = 0:1023;

testData = StokesData(testSamples, u_bc);
[~, ~, ~, meanSqDist, ~, mll] =...
    rom.predict(testData, 'local');

save('./prediction.mat', 'meanSqDist', 'mll');