%Script to create gif animation of prediction
clear
%close all
addpath('~/matlab/toolboxes/gif_v1.0/gif');
load('./workspace.mat');
close all;

nStart = 0;
phi_Iter = .5;
nEnd = 360;
sample = 2;

f = figure('units', 'normalized', 'outerposition', [0 0 1 1]);
s1 = subplot(1, 1, 1);

%data
ts = trisurf(testData.cells{samples(sample)},...
    testData.X{samples(sample)}(:, 1),...
    testData.X{samples(sample)}(:, 2), testData.P{samples(sample)},...
    'LineStyle', 'none');
axis tight;
xticks({});
yticks({});
zticks({});
hold on;
%mean
s = surf(reshape(testData.X_interp{1}(:,1), 129, 129),...
    reshape(testData.X_interp{1}(:,2), 129, 129),...
    reshape(predMean{samples(sample)}, 129, 129), 'linestyle', 'none',...
    'facecolor', 'b');

%mean + .5s td
mps = surf(reshape(testData.X_interp{1}(:,1), 129, 129),...
    reshape(testData.X_interp{1}(:,2), 129, 129),...
    reshape(predMean{samples(sample)}, 129, 129) +...
    .5*reshape(sqrt(predVar{samples(sample)}), 129, 129), 'linestyle', 'none',...
    'facecolor', [.85 .85 .85], 'facealpha', .7);

%mean + .5s td
mms = surf(reshape(testData.X_interp{1}(:,1), 129, 129),...
    reshape(testData.X_interp{1}(:,2), 129, 129),...
    reshape(predMean{samples(sample)}, 129, 129) -...
    .5*reshape(sqrt(predVar{samples(sample)}), 129, 129), 'linestyle', 'none',...
    'facecolor', [.85 .85 .85], 'facealpha', .7);


outerpos = s1.OuterPosition;
ti = s1.TightInset;
left = outerpos(1) + ti(1);
bottom = outerpos(2) + ti(2);
ax_width = outerpos(3) - ti(1) - ti(3);
ax_height = outerpos(4) - ti(2) - ti(4);
s1.Position = [left bottom ax_width ax_height];


for phi = nStart:phi_Iter:nEnd    
    view(phi, 15);
    drawnow
    if phi == nStart
        %create gif
        gif('predRotation3.gif', 'DelayTime', .02, 'frame', f);
    else
        gif
    end
end


