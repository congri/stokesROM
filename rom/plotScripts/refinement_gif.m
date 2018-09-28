%% Plot script to generate gif animation of mesh refinement
%% Preamble
clear;
addpath('./mesh')
addpath('./FEM')
addpath('~/matlab/toolboxes/gif_v1.0/gif')
fig = figure('units','normalized','outerposition',[0 0 .55 1]);
sp1 = subplot(1, 1, 1, 'Parent', fig);
filename = 'reduced_elbo_refine';


%% Load data, set constants
score = dlmread('./data/cell_score');
coarseGridX = (1/16)*ones(1, 16);
coarseGridY = (1/16)*ones(1, 16);

%% Find first rf2fem
curr_score = score(1, :);
curr_score = curr_score(curr_score ~= 0);
if numel(curr_score) == 4
    gridRF = RectangularMesh((1/2)*ones(1, 2));
elseif numel(curr_score) == 16
    gridRF = RectangularMesh((1/4)*ones(1, 4));
else
    error('What initial mesh size?')
end
cell_dictionary = 1:gridRF.nCells;
rf2fem = gridRF.map2fine(coarseGridX, coarseGridY);

for iter = 2:size(score, 1)
    %loop over all iterations
    curr_score = score(iter - 1, :);
    curr_score = curr_score(curr_score ~= 0);
    next_score = score(iter, :);
    next_score = next_score(next_score ~= 0);
    
    if numel(curr_score) ~= numel(next_score)
        [~, cell_index_pde] = min(curr_score)
        splt_cll = find(cell_dictionary == cell_index_pde)
        gridRF.split_cell(gridRF.cells{splt_cll});
        rf2fem = gridRF.map2fine(coarseGridX, coarseGridY);
        
        %Update cell index dictionary
        cell_dictionary(splt_cll) = nan;
        cell_dictionary((splt_cll + 1):end) = ...
            cell_dictionary((splt_cll + 1):end) - 1;
        if isnan(cell_dictionary(end))
            cell_dictionary = [cell_dictionary,(cell_dictionary(end - 1)+ 1):...
                (cell_dictionary(end - 1) + 4)];
        else
            cell_dictionary = [cell_dictionary, (cell_dictionary(end) + 1):...
                (cell_dictionary(end) + 4)];
        end
    end
    
    %% Plot
    imagesc(reshape(rf2fem*(-next_score'), numel(coarseGridX),...
        numel(coarseGridY))', 'Parent', sp1);
    sp1.GridLineStyle = 'none';
    sp1.XTick = [];
    sp1.YTick = [];
    sp1.Title.String = 'Elbo cell score';
    if iter == 2
        %create gif
        gif(strcat(filename, '.gif'), 'DelayTime', .02, 'frame', fig);
    else
        gif
    end
    drawnow
end