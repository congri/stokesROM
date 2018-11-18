%Plots mesh underneath prediction

f = gcf;
test_sample = 256;
resolution = 1024;
z1 = [-4e5 -3e6 -5e5 -3e5 -1e5 -1e5];
z2 = [6e5 3e6 4.5e5 4.5e5 1e5 1.5e5];
for i = 1:6
    filename = '/home/constantin/cluster/python/data/stokesEquation/meshSize=256/nonOverlappingDisks/margins=0.003_0.003_0.003_0.003/N~logn/mu=8.35/sigma=0.6/x~GP/cov=squaredExponential/l=0.1/sig_scale=1.5/r~logn/mu=-5.53/sigma=0.3/microstructureInformation';
    filename = strcat(filename, num2str(test_sample), '.mat');
    load(filename);
    test_sample = test_sample + 1;
    ax = f.Children(14 - 2*i);
    ax.ZLim(1) = z1(i);
    ax.ZLim(2) = z2(i);
    
    [xx, yy] = meshgrid(linspace(0, 1, resolution));
    r2 = diskRadii.^2;
    img = false(resolution);
    
    for n = 1:numel(diskRadii)
        img = img | ((xx - diskCenters(n, 1)).^2 + (yy - diskCenters(n, 2)).^2 ...
            <= r2(n));
    end
    img_Z = ax.ZLim(1)*ones(size(img));
    img_C = ax.CLim(2)*img;
    img_C(img_C == 0) = ax.CLim(1);
    s = surf(xx, yy, img_Z, img_C, 'Parent', ax, 'linestyle', 'none');
    
    ax.XTick = [.25 .5 .75];
    ax.YTick = [.25 .5 .75];
    ax.XTickLabel = {};
    ax.YTickLabel = {};
    ax.GridLineStyle = '-';
    ax.BoxStyle = 'back';

end