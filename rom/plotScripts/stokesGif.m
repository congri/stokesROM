%Script to create gif animation of random stokes data
clear
%close all
addpath('~/matlab/toolboxes/gif_v1.0/gif');
addpath('~/matlab/toolboxes/plotting/export_fig');

filename = 'randDarcyData';

stokesData = StokesData(0:127);

nStart = 1;
nEnd = 8;

max_c = 500;
min_c = 0;
lambda_f = mfile.cond(:, nStart:nEnd);
max_l = max(max(lambda_f));
min_l = min(min(lambda_f));
%lambda_f colorscale transformation
k = (max_c - min_c)/(max_l - min_l);
c = max_c - max_l*k;
lambda_f = k*lambda_f + c;

[xx, yy] = meshgrid(1:257);

f = figure('units', 'normalized', 'outerposition', [0 0 1 1]);
iter = 0;
for n = nStart:nEnd
    s1 = subplot(1, 1, 1);
    
    hold off;
    sp = surf(xx, yy, reshape(u_f(:, n), 257, 257), 'linestyle', 'none');
    hold on;
    img = imagesc(reshape(lambda_f(:, n), 256, 256));
    
%     s1.BoxStyle = 'back';%axes on back side
%     s1.Visible = 'off';  %invisible axes
%     view(2);
    xticks({});
    yticks({});
    zticks({});
    s1.ZLim = [min_c, max_c];
    s1.XLim = [1 257];
    s1.YLim = [1 257];
    caxis(s1.ZLim);
    view(-25, 15)
    s1.Title.String = '$p$';
    s1.Title.FontSize = 40;
    drawnow;
       
%     s2 = subplot(1, 2, 2);
%     ts = trisurf(cells, x(:, 1), x(:, 2), p, 'LineStyle', 'none', 'Parent', s2);
%     s2.Title.String = '$p$';
%     s2.Title.FontSize = 46;
%     s2.BoxStyle = 'back';%axes on back side
% %     s2.Visible = 'off';  %invisible axes
%     xticks({});
%     yticks({});
%     zticks({});
%     min_p = min(p)
%     max_p = max(p)
%     axis tight
% %     s2.ZLim = [-1e3, 3e4];
%     view(-45, 15);
    
    savegif = false;
    if savegif
        if n == nStart
            %create gif
            gif(strcat(filename, '.gif'), 'DelayTime', .45, 'frame', f);
        else
            gif
        end
    end
    %save frames
    saveframe = true;
    if saveframe
        export_fig(strcat('./frames_Darcy/', filename, num2str(iter)),...
            '-png');       
    end
    iter = iter + 1;
    end