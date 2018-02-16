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

rom = StokesROM;
testSamples = 128:135;
testData = StokesData(testSamples, u_bc);
[~, ~, effDiff] = rom.predict(testData, 'local');