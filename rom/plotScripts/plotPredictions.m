%plot predictions vs. real unsmoothed data

samples = 2:5;

testData.readData('xpc');
testData.shiftData(false);

N = numel(samples);
f = figure('units','normalized','outerposition',[0 0 1 1]);
for n = 1:N
    sp(n) = subplot(1, N, n);
    
    %data
    ts(n) = trisurf(testData.cells{samples(n)}, testData.X{samples(n)}(:, 1),...
        testData.X{samples(n)}(:, 2), testData.P{samples(n)},...
        'LineStyle', 'none');
    axis tight;
    sp(n).ZLim = [-24750, 4146];
    xticks({});
    yticks({});
    zticks({});
    hold on;
    
    %mean
    s(n) = surf(reshape(testData.X_interp{1}(:,1), 129, 129),...
        reshape(testData.X_interp{1}(:,2), 129, 129),...
        reshape(predMean{samples(n)}, 129, 129), 'linestyle', 'none',...
        'facecolor', 'b');
    
    %mean + .5s td
    mps(n) = surf(reshape(testData.X_interp{1}(:,1), 129, 129),...
        reshape(testData.X_interp{1}(:,2), 129, 129),...
        reshape(predMean{samples(n)}, 129, 129) +...
        .5*reshape(sqrt(predVar{samples(n)}), 129, 129), 'linestyle', 'none',...
        'facecolor', [.85 .85 .85], 'facealpha', .7);
    
    %mean + .5s td
    mms(n) = surf(reshape(testData.X_interp{1}(:,1), 129, 129),...
        reshape(testData.X_interp{1}(:,2), 129, 129),...
        reshape(predMean{samples(n)}, 129, 129) -...
        .5*reshape(sqrt(predVar{samples(n)}), 129, 129), 'linestyle', 'none',...
        'facecolor', [.85 .85 .85], 'facealpha', .7);
end
export_fig('./predictions', '-png', '-r300');



%Script to create gif animation of random stokes data
clear
%close all
addpath('~/matlab/toolboxes/gif_v1.0/gif');

foldername = strcat('~/cluster/python/data/stokesEquation/meshes/',...
    'meshSize=128/nNonOverlapCircExcl=logn5.0-1.1/',...
    'coordDist=gauss_mu=[0.5, 0.5]cov=[[0.035, 0.0], [0.0, 0.8]]',...
    '_margins=(-1, 0.01, -1, 0.01)/radiiDist=logn_r_params=',...
    '(-4.0, 0.7)/');
foldername_solution = strcat(foldername,...
    'u_x=1.0_u_y=0.0/');

nStart = 0;
nEnd = 24;

f = figure('units', 'normalized', 'outerposition', [0 0 1 1]);
for n = nStart:nEnd
    mesh_file = strcat(foldername, 'mesh', num2str(n), '.mat');
    load(mesh_file);
    solution_file = strcat(foldername_solution, 'solution', num2str(n), '.mat');
    load(solution_file);
    p_temp = p';
    p_origin = p_temp(all((x == [0, 0])'));
    p_temp = p_temp - p_origin;
    p = p_temp';
    
    s1 = subplot(1, 2, 1);
%     s1.ZLim = [-13500, 12200];
    ts = trisurf(cells, x(:, 1), x(:, 2), sqrt(sum(u.^2)),...
        'LineStyle', 'none', 'Parent', s1);
    s1.Title.String = '$|\mathbf v|$';
    s1.Title.FontSize = 42;
%     s1.BoxStyle = 'back';%axes on back side
%     s1.Visible = 'off';  %invisible axes
    view(2);
    xticks({});
    yticks({});
    zticks({});
    
    outerpos = s1.OuterPosition;
    ti = s1.TightInset;
    left = outerpos(1) + ti(1);
    bottom = outerpos(2) + ti(2);
    ax_width = outerpos(3) - ti(1) - ti(3);
    ax_height = outerpos(4) - ti(2) - ti(4);
    s1.Position = [left bottom ax_width ax_height];
    drawnow

    
    s2 = subplot(1, 2, 2);
    ts = trisurf(cells, x(:, 1), x(:, 2), p, 'LineStyle', 'none', 'Parent', s2);
    s2.Title.String = '$p$';
    s2.Title.FontSize = 46;
    s2.BoxStyle = 'back';%axes on back side
%     s2.Visible = 'off';  %invisible axes
    xticks({});
    yticks({});
    zticks({});
    min_p = min(p)
    max_p = max(p)
    axis tight
%     s2.ZLim = [-1e3, 3e4];
    view(-45, 15);
    
    outerpos = s2.OuterPosition;
    ti = s2.TightInset;
    left = outerpos(1) + ti(1);
    bottom = outerpos(2) + ti(2);
    ax_width = outerpos(3) - ti(1) - ti(3);
    ax_height = outerpos(4) - ti(2) - ti(4);
    s2.Position = [left bottom ax_width ax_height];
    drawnow
    if n == nStart
        %create gif
        gif('randStokesData.gif', 'DelayTime', .45, 'frame', f);
    else
        gif
    end
end