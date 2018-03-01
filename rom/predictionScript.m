%prediction script
addpath('./mesh')
addpath('./featureFunctions/nonOverlappingPolydisperseSpheres')

%Boundary condition fields
p_bc = @(x) 0;
%influx?
u_bc{1} = 'u_x=0.25 - (x[1] - 0.5)*(x[1] - 0.5)';
u_bc{2} = 'u_y=0.0';
u_bc{1} = 'u_x=-0.8 + 2.0*x[1]';
u_bc{2} = 'u_y=-1.2 + 2.0*x[0]';
u_bc{1} = 'u_x=1.0';
u_bc{2} = 'u_y=0.0';

rom = StokesROM;
testSamples = 9:15;
testData = StokesData(testSamples, u_bc);
[~, ~, effDiff, meanSqDist, sqDist] = rom.predict(testData, 'local');