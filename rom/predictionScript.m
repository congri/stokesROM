%prediction script

%coarse FEM mesh grid vectors
gridX = (1/4)*ones(1, 4);
gridY = (1/4)*ones(1, 4);

%p_cf variance grid vectors
gridSX = [.125, .25, .5, ones(1, 26), .5, .25, .125];   %p_cf S grid
gridSX = gridSX/sum(gridSX);
gridSY = gridSX;

%Boundary condition fields
p_bc = @(x) 0;
%influx?
% u_bc{1} = @(x) 0;        %lower bound
% u_bc{2} = @(y) 0.25 - (y - 0.5)*(y - 0.5);         %right bound
% u_bc{3} = @(x) 0;         %upper bound
% u_bc{4} = @(y) -0.25 + (y - 0.5)*(y - 0.5);        %left bound
u_bc{1} = 'u_x=0.25 - (x[1] - 0.5)*(x[1] - 0.5)';
u_bc{2} = 'u_y=0.0';

rom = StokesROM;
testSamples = 128:135;
testData = StokesData(testSamples, u_bc);
[~, ~, effDiff] = rom.predict(testData, 'local');