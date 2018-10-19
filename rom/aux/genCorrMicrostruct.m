%% Script to generate list of circular exclusions distributed according to
% a warped Gaussian process
clear;

lengthScale = .1;
covarianceFunction = 'squaredExponential';
sigmoid_scale = 1.5;         %sigmoid warping length scale param
nBochnerSamples = 5000;
nExclusionParams = [8.35, .6];
margins = [.003, .003, .003, .003];
rParams = [-5.53, 0.3];
nMeshes = 79:2500;
t_max = 3600;                 %in seconds
plt = false;

addpath('~/cluster/matlab/projects/rom/genConductivity');
if plt
    f = figure;
end

%% Set up save path
savepath = '~/cluster/python/data/stokesEquation/meshSize=256/nonOverlappingDisks/margins=';
savepath = strcat(savepath, num2str(margins(1)), '_', num2str(margins(2)),...
    '_', num2str(margins(3)), '_', num2str(margins(4)), '/N~logn/mu=', ...
    num2str(nExclusionParams(1)), '/sigma=', num2str(nExclusionParams(2)), ...
    '/x~GP/cov=', covarianceFunction, '/l=', num2str(lengthScale),...
    '/sig_scale=', num2str(sigmoid_scale), '/r~logn/mu=', num2str(rParams(1)), ...
    '/sigma=', num2str(rParams(2)));

if ~exist(savepath, 'dir')
    mkdir(savepath);
end

mesh_iter = 1;
for n = nMeshes
    sampleFun = genBochnerSamples(lengthScale,1,nBochnerSamples,covarianceFunction);
    sampleFun = @(x) sigmf(sampleFun(x), [sigmoid_scale, 0]);
    
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
        if(rand <= sampleFun(diskCenter'))
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
            subplot(2, nCols, mesh_iter, 'Parent', f);
            [xx, yy] = meshgrid(linspace(0, 1, 101));
            x = [xx(:) yy(:)]';
            zz = reshape(sampleFun(x), 101, 101);
            sf = surf(xx, yy, zz);
            xticks([]);
            yticks([]);
            view(2);
            sf.LineStyle = 'none';
            
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







