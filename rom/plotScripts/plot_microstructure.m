%% script to generate plot of microstructural data (independent of mesh)

%% params
resolution = 2048;

%% load data
%load('/home/constantin/python/data/stokesEquation/meshSize=256/nonOverlappingDisks/margins=0.003_0.003_0.003_0.003/N~logn/mu=8.35/sigma=0.6/x~GP/cov=squaredExponential/l=0.1/sig_scale=1.5/r~logn/mu=-5.53/sigma=0.3/microstructureInformation2.mat')




[xx, yy] = meshgrid(linspace(0, 1, resolution));
x = [xx(:) yy(:)];
clear xx yy;

img = true(resolution);

%loop over every pixel and check if solid or fluid
for i = 1:size(x, 1)
    distSq = sum((diskCenters - x(i, :)).^2, 2);
    if any(distSq' <= diskRadii.^2)
        %inside of exclusion, i.e. outside of domain
        img(i) = false;
    end
end

% f = figure;
img_h = imagesc(img);






