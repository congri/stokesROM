%Plots samples of Stokes flow with file names usable as LaTeX frames
clear;
% close all;
samples = 0;
stokesData = StokesData(samples);

%Load data to make sure that not interpolated data is plotted
stokesData.readData('c');
stokesData.readData('x');
stokesData.readData('p');
stokesData.readData('u');

tumblau = [0 101/255 189/255];


max_p = -Inf;
for n = 1:numel(samples)
    if(max_p < (max(stokesData.P{n}) - min(stokesData.P{n})))
        max_p = (max(stokesData.P{n}) - min(stokesData.P{n}));
    end
end

pltIndex = 1;
for n = 1:numel(samples)
    %figure(figHandle);
    figHandle(pltIndex) = figure('units','normalized',...
        'outerposition',[0 0 1 1]);;
    
%     %flow bc arrow sketch
%     %a lot by hand here!
%     dx = [.039 0];
%     x = [.1 .1555];
%     u0 = [.03, .058];
%     u1 = u0;
%     n_arr = 8;
%     for i = 1:n_arr
%         an(i) = annotation('arrow', [x(1) x(1) + u1(1)], [x(2)   x(2) + u1(2)]);
%         an(i).Color = tumblau;
%         an(i).LineWidth = 6;
%         an(i).HeadLength = 12;
%         an(i).HeadWidth = 16;
%         
%         x = x + dx;
%         u1 = u1 - 2*(u0/(n_arr - 1));
%     end
%     
%     %flow bc arrow sketch
%     %a lot by hand here!
%     dx = [.038 -.0025];
%     x = [.225 .4];
%     u0 = [-.03, .058];
%     u1 = u0;
%     n_arr = 8;
%     for i = 1:n_arr
%         an2(i) = annotation('arrow', [x(1) x(1) + u1(1)], [x(2)   x(2) + u1(2)]);
%         an2(i).Color = tumblau;
%         an2(i).LineWidth = 6;
%         an2(i).HeadLength = 12;
%         an2(i).HeadWidth = 16;
%         
%         x = x + dx;
%         u1(2) = u1(2) - 1.75*(u0(2)/(n_arr - 1));
%         u1 = 1.03*u1;
%     end
    
    %Mesh
    ax_handle(1, pltIndex) = subplot(1, 2, 1);
    microstruct_handle(1, pltIndex) = trisurf(stokesData.cells{n},...
        stokesData.X{n}(:, 1), stokesData.X{n}(:, 2),...
        zeros(size(stokesData.X{n}, 1),1), 'linewidth', .5);
    microstruct_handle(1, pltIndex).FaceColor = [0 0 0];
    hold(ax_handle(1, pltIndex), 'on');
    
    
    %pressure field
    P_shift = stokesData.P{n} - min(stokesData.P{n});
    p_handle(2, pltIndex) =...
        trisurf(stokesData.cells{n}, stokesData.X{n}(:, 1),...
        stokesData.X{n}(:, 2), P_shift);
    p_handle(2, pltIndex).LineStyle = 'none';
    axis square;
    axis tight;
    phi = 20;
    view(phi, 20);
    grid off;
    box on;
    xticks({});
    yticks({});
    ax_handle(1, pltIndex).ZLim = [0, max_p];
    
    
    
    %velocity field (norm)
    u_norm = sqrt(sum(stokesData.U{n}.^2));

    %velocity field (norm), 2d
    ax_handle(2, pltIndex) = subplot(1, 2, 2);
    p_handle(3, pltIndex) = trisurf(stokesData.cells{n},...
        stokesData.X{n}(:, 1), stokesData.X{n}(:, 2), u_norm);
    p_handle(3, pltIndex).LineStyle = 'none';
    axis square;
    axis tight;
    view(2);
    grid off;
    box on;
    xticks({});
    yticks({});
    
    ax_handle(1, pltIndex).Color = [.95 .95 .95];
    pltIndex = pltIndex + 1;
end










