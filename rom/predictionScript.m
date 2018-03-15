%prediction script
addpath('./mesh')
addpath('./featureFunctions/nonOverlappingPolydisperseSpheres')

%Boundary condition fields
p_bc = @(x) 0;
%influx?
u_bc{1} = 'u_x=1.0';
u_bc{2} = 'u_y=0.0';
u_bc{1} = 'u_x=-0.8 + 2.0*x[1]';
u_bc{2} = 'u_y=-1.2 + 2.0*x[0]';

%rom = StokesROM;
testSamples = 2:7;

testData = StokesData(testSamples, u_bc);
[predMean, predVar, effDiff, meanSqDist, sqDist, mll] =...
    rom.predict(testData, 'local');
