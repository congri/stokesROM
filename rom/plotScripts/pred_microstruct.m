%Plots mesh underneath prediction

f = gcf;
test_sample = [512 512 512 512];
resolution = 1024;
% z1 = [0 0 0 0 0 0];
% z2 = [2.5e5 4.2e5 2e5 1.2e5 4e5 4.5e5];
% z1 = [-5e4 -7e4 -1e4];
% z2 = [1.5e4 5e4 4e5];
for i = 1:4
    filename = '/home/constantin/cluster/python/data/stokesEquation/meshSize=256/nonOverlappingDisks/margins=0.003_0.003_0.003_0.003/N~logn/mu=7.8/sigma=0.2/x~GP/cov=squaredExponential/l=0.08/sig_scale=1.2/r~lognGP/mu=-5.23/sigma=0.3/sigmaGP_r=0.4/l=0.05/microstructureInformation';
    filename = strcat(filename, num2str(test_sample(i)), '.mat');
    load(filename);
    ax = f.Children(5 - i);
%     ax.ZLim(1) = z1(i);
%     ax.ZLim(2) = z2(i);
    
    [xx, yy] = meshgrid(linspace(0, 1, resolution));
    r2 = diskRadii.^2;
    img = false(resolution);
    
    for n = 1:numel(diskRadii)
        img = img | ((xx - diskCenters(n, 1)).^2 + (yy - diskCenters(n, 2)).^2 ...
            <= r2(n));
    end
    img = ~img;
    img_Z = ax.ZLim(1)*ones(size(img));
    img_C = ax.CLim(2)*img;
    img_C(img_C == 0) = ax.CLim(1);
    s = surf(xx, yy, img_Z, img_C, 'Parent', ax, 'linestyle', 'none');
    
%     ax.XTick = [.25 .5 .75];
%     ax.YTick = [.25 .5 .75];
%     ax.XTickLabel = {};
%     ax.YTickLabel = {};
%     ax.GridLineStyle = '-';
%     ax.BoxStyle = 'full';
drawnow;
end