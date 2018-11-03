%% Script to generate list of circular exclusions distributed according to
% a warped Gaussian process
clear;

mode = 'tiles';    %engineered or GP

lengthScale = .1;
covarianceFunction = 'squaredExponential';
sigmoid_scale = 1.5;         %sigmoid warping length scale param
nBochnerSamples = 5000;
nExclusionParams = [8.1, .6];
margins = [.003, .003, .003, .003];
rParams = [-5.53, 0.3];
nMeshes = 0:2500;
t_max = 3600;                 %in seconds
plt = true;

addpath('~/cluster/matlab/projects/rom/genConductivity');
if plt
    f = figure;
end

%% Set up save path
if strcmp(mode, 'GP')
    savepath = '~/cluster/python/data/stokesEquation/meshSize=256/nonOverlappingDisks/margins=';
    savepath = strcat(savepath, num2str(margins(1)), '_', num2str(margins(2)),...
        '_', num2str(margins(3)), '_', num2str(margins(4)), '/N~logn/mu=', ...
        num2str(nExclusionParams(1)), '/sigma=', num2str(nExclusionParams(2)), ...
        '/x~GP/cov=', covarianceFunction, '/l=', num2str(lengthScale),...
        '/sig_scale=', num2str(sigmoid_scale), '/r~logn/mu=', num2str(rParams(1)), ...
        '/sigma=', num2str(rParams(2)));
elseif strcmp(mode, 'engineered')
    savepath = '~/cluster/python/data/stokesEquation/meshSize=256/nonOverlappingDisks/margins=';
    savepath = strcat(savepath, num2str(margins(1)), '_', num2str(margins(2)),...
        '_', num2str(margins(3)), '_', num2str(margins(4)), '/N~logn/mu=', ...
        num2str(nExclusionParams(1)), '/sigma=', num2str(nExclusionParams(2)), ...
        '/x~engineered','/r~logn/mu=', num2str(rParams(1)), ...
        '/sigma=', num2str(rParams(2)));
elseif strcmp(mode, 'tiles')
    savepath = '~/cluster/python/data/stokesEquation/meshSize=256/nonOverlappingDisks/margins=';
    savepath = strcat(savepath, num2str(margins(1)), '_', num2str(margins(2)),...
        '_', num2str(margins(3)), '_', num2str(margins(4)), '/N~logn/mu=', ...
        num2str(nExclusionParams(1)), '/sigma=', num2str(nExclusionParams(2)), ...
        '/x~tiles','/r~logn/mu=', num2str(rParams(1)), ...
        '/sigma=', num2str(rParams(2)));
else
    error('unknown mode');
end

if ~exist(savepath, 'dir')
    mkdir(savepath);
end

mesh_iter = 1;
for n = nMeshes
    if strcmp(mode, 'GP')
        rejectionFun = genBochnerSamples(lengthScale,1,nBochnerSamples,covarianceFunction);
        rejectionFun = @(x) sigmf(rejectionFun(x), [sigmoid_scale, 0]);
    elseif strcmp(mode, 'engineered')
        rejectionFun = @(x) engineeredRejectionFun(x);
    elseif strcmp(mode, 'tiles')
        rejections = get_rejections(8);
        rejectionFun = @(x) tilesRejectionFun(x, rejections);
    else
        error('unknown mode')
    end
    
    nExclusions = round(lognrnd(nExclusionParams(1), nExclusionParams(2)))
    diskCenters = zeros(nExclusions, 2);
    diskRadii = zeros(1, nExclusions);
    currentDisks = 0;
    
    t0 = tic;
    t_elapsed = 0;
    while(currentDisks < nExclusions && t_elapsed < t_max)
        diskCenter = rand(1, 2);
        diskRadius = lognrnd(rParams(1), rParams(2));
        
        %accept/reject
        if(rand <= rejectionFun(diskCenter'))
            %check if new circle overlaps with another circle
            
            overlap = false;
            if currentDisks     %first disk cannot overlap            
                overlap = any(((diskRadius + diskRadii(1:currentDisks))').^2 >=...
                    sum((diskCenter - diskCenters(1:currentDisks, :)).^2, 2));
            end
            
            if ~overlap
                onBoundary = false;
                if(((diskCenter(2) - diskRadius) < margins(1) &&...
                        margins(1) >= 0) || ...
                        (((diskCenter(1) + diskRadius) > 1- margins(2))...
                        && margins(2) >= 0) ||...
                        (((diskCenter(2) + diskRadius) > 1 - margins(3))...
                        && margins(3) >= 0) ||...
                        ((diskCenter(1) - diskRadius) < margins(4))...
                        && (margins(4) >= 0))
                    onBoundary = true;
                end
                if ~onBoundary
                    diskCenters(currentDisks + 1, :) = diskCenter;
                    diskRadii(currentDisks + 1) = diskRadius;
                    currentDisks = currentDisks + 1;
                end
            end
        end
        t_elapsed = toc(t0);
    end
    t_elapsed
    
    if currentDisks < nExclusions
        diskCenters((currentDisks + 1):end, :) = [];
        diskRadii((currentDisks + 1):end) = [];
    end
    
    %% save microstructure data
    save(strcat(savepath, '/microstructureInformation_nomesh', num2str(n)),...
        'diskCenters', 'diskRadii');
    
    %% Plotting
    if plt
        nCols = min([5, nMeshes(end) + 1]);
        hold on;
        
        if mesh_iter < nCols
            if strcmp(mode, 'GP')
                subplot(2, nCols, mesh_iter, 'Parent', f);
                [xx, yy] = meshgrid(linspace(0, 1, 101));
                x = [xx(:) yy(:)]';
                zz = reshape(rejectionFun(x), 101, 101);
                sf = surf(xx, yy, zz);
                xticks([]);
                yticks([]);
                view(2);
                sf.LineStyle = 'none';
            end
            
            subplot(2, nCols, mesh_iter + nCols, 'Parent', f);
            p = plot(diskCenters(:, 1), diskCenters(:, 2), 'ko');
            xticks([]);
            yticks([]);
            p.LineWidth = .5;
            p.MarkerSize = 2;
        end
    end
    mesh_iter = mesh_iter + 1;
end


function [r] = engineeredRejectionFun(x)
    %[.0 .0]-[.25 .25] cell is empty
    if(x(1) <= .25 && x(2) <= .25)
        r = 0;
    elseif(x(1) > .25 && x(1) <= .5 && x(2) <= .25)
        r = 4*x(1) - 1;
    elseif(x(2) > .25 && x(2) <= .5 && x(1) <= .25)
        r = 4*x(2) - 1;
    elseif(x(1) > .25 && x(1) <= .5 && x(2) > .25 && x(2) <= .5)
        r = -3 + 8*x(1) + 8*x(2) - 16*x(1)*x(2);
    elseif(x(1) > .5 || x(2) > .5)
        r = 1;
    else
        r = 0;
    end

end


function [r] = tilesRejectionFun(x, rejections)
    %[.0 .0]-[.25 .25] cell is empty
    N = sqrt(numel(rejections));
    [xx, yy] = meshgrid(linspace(0, 1, N + 1));
    xxl = xx(1:N, 1:N);
    yyl = yy(1:N, 1:N);
    xxu = xx(1:N, 2:(N + 1));
    yyu = yy(2:(N + 1), 1:N);
    l = [xxl(:), yyl(:)];
    u = [xxu(:), yyu(:)];

    cell = find((x(1) >= l(:, 1)).*(x(2) >= l(:, 2)).*(x(1) < u(:, 1)).*(x(2) < u(:, 2)));
    r = rejections(cell);
end


function [r] = get_rejections(N)
    %[.0 .0]-[.25 .25] cell is empty
    %N is the linear cell number i.e. 8 for an 8x8 mesh
    [xx, yy] = meshgrid(linspace(0, 1, N + 1));
    xxu = xx(1:N, 2:(N + 1));
    yyu = yy(2:(N + 1), 1:N);
    u = [xxu(:), yyu(:)];
    
    r = .5*ones(64, 1);
    
    for i = 1:64
        if(u(i, 1) <= .5 && u(i, 2) <= .5)
            %lower left square
            r(i) = rand;
        end
    end
end















