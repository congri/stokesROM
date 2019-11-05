%Fake effective solution plot constructed from finescale data plot


effSolution = true;
if effSolution
    f = gcf;
    for i = 1:4
        s2 = subplot(2, 4, i);
        
        truesol = f.Children(end - i + 1).Children(6).ZData;
        
        %cut out 4x4
        truesol(98:128, :) = [];
        truesol(66:96, :) = [];
        truesol(34:64, :) = [];
        truesol(2:32, :) = [];
        truesol(:, 98:128) = [];
        truesol(:, 66:96) = [];
        truesol(:, 34:64) = [];
        truesol(:, 2:32) = [];
        
        [X, Y] = meshgrid(linspace(0, 1, 5));
        [Xq, Yq] = meshgrid(linspace(0, 1, 257));
        Vq = interp2(X, Y, truesol, Xq, Yq, 'nearest');
        
        sf = surf(Xq, Yq, Vq, 'linestyle', 'none', 'Parent', s2);
        ax = gca;
        ax.XTick = [0 .25 .5 .75 1];
        ax.XTickLabel = [];
        ax.YTick = [0 .25 .5 .75 1];
        ax.YTickLabel = [];
        ax.ZTickLabel = [];
        hold(s2, 'on');
        p = plot3(linspace(0, 1, 257), .125*ones(1, 257), Vq(33, :), 'linewidth', 1, ...
            'color', 'k');
        p = plot3(linspace(0, 1, 257), .375*ones(1, 257), Vq(97, :), 'linewidth', 1, ...
            'color', 'k');
        p = plot3(linspace(0, 1, 257), .625*ones(1, 257), Vq(161, :), 'linewidth', 1, ...
            'color', 'k');
        p = plot3(linspace(0, 1, 257), .875*ones(1, 257), Vq(225, :), 'linewidth', 1, ...
            'color', 'k');
        p = plot3(.125*ones(1, 257), linspace(0, 1, 257), Vq(:, 33), 'linewidth', 1, ...
            'color', 'k');
        p = plot3(.375*ones(1, 257), linspace(0, 1, 257), Vq(:, 97), 'linewidth', 1, ...
            'color', 'k');
        p = plot3(.625*ones(1, 257), linspace(0, 1, 257), Vq(:, 161), 'linewidth', 1, ...
            'color', 'k');
        p = plot3(.875*ones(1, 257), linspace(0, 1, 257), Vq(:, 225), 'linewidth', 1, ...
            'color', 'k');
        axis tight
        hold(s2, 'off')
        
%         if i == 1
%             gif(strcat('~/cluster/images/multModHub/randBC', '.gif'),...
%                 'DelayTime', 2, 'frame', f2);
%         else
%             gif;
%         end
    end
end