%prediction script
addpath('./mesh')
addpath('./featureFunctions/nonOverlappingPolydisperseSpheres')


%coarse FEM mesh grid vectors
gridX = (1/4)*ones(1, 4);
gridY = (1/4)*ones(1, 4);

%Boundary condition fields
p_bc = @(x) 0;
%influx?
u_bc{1} = 'u_x=0.25 - (x[1] - 0.5)*(x[1] - 0.5)';
u_bc{2} = 'u_y=0.0';
u_bc{1} = 'u_x=-0.8 + 2.0*x[1]';
u_bc{2} = 'u_y=-1.2 + 2.0*x[0]';

rom = StokesROM;
testSamples = 0:5;
testData = StokesData(testSamples, u_bc);
[~, ~, effDiff, meanSqDist, sqDist] = rom.predict(testData, 'local');