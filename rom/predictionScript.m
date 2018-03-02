%prediction script
addpath('./mesh')
addpath('./featureFunctions/nonOverlappingPolydisperseSpheres')

%Boundary condition fields
p_bc = @(x) 0;
%influx?
u_bc{1} = 'u_x=1.0';
u_bc{2} = 'u_y=0.0';

rom = StokesROM;
testSamples = 33:38;
testData = StokesData(testSamples, u_bc);
[predMean, predVar, effDiff, meanSqDist, sqDist] =...
    rom.predict(testData, 'local');