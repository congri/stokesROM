%% Plot script to generate gif animation of mesh refinement
%% Preamble
clear;
addpath('./mesh')
addpath('./FEM')
addpath('~/matlab/toolboxes/gif_v1.0/gif')
fig = figure('units','normalized','outerposition',[0.2 0.2 .115 .25]);
fig.ToolBar = 'none';
sp1 = subplot(1, 1, 1, 'Parent', fig);
sp1.Position(1) = .015;
sp1.Position(2) = -.01;
sp1.Position(4) = .9;
sp1.Position(3) = .95;
fig.Position(4) = .22;
refinement_mode = 'activeCells';
filename = strcat(refinement_mode, '_refine');
seconds_per_frame = .05;
iter_increment = 1;
plot_split_only = false;
if plot_split_only
    seconds_per_frame = 2;
    filename = strcat(filename, '_split_only');
end


%% Load data, set constants
if strcmp(refinement_mode, 'random')
    score_temp = dlmread('./data/cell_score');
    score = rand(size(score_temp));
    score(score_temp == 0) = 0;
else
    score = dlmread(strcat('./data/', refinement_mode));
end
coarseGridX = (1/16)*ones(1, 16);
coarseGridY = (1/16)*ones(1, 16);
gridFEM = RectangularMesh(coarseGridX);
load('./data/modelParams.mat');

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
rf2fem = gridRF.map2fine(gridFEM);
%% Plot first frame
next_score = score(1, :);
next_score = next_score(next_score ~= 0);
if(strcmp(refinement_mode, 'cell_score') || ...
        strcmp(refinement_mode, 'cell_score_full'))
    next_score = -next_score;
end
imagesc(reshape(rf2fem*(next_score'), numel(coarseGridX),...
    numel(coarseGridY))', 'Parent', sp1);
sp1.GridLineStyle = 'none';
sp1.XTick = [];
sp1.YTick = [];
sp1.Title.String = 'Elbo cell score';
sp1.YDir = 'normal';
gif(strcat(filename, '.gif'), 'DelayTime', seconds_per_frame, 'frame', fig);
axis tight
drawnow

if plot_split_only
    plt_iter = false;
else
    plt_iter = true;
end

curr_split = 1;
for iter = 2:iter_increment:size(score, 1)
    %loop over all iterations
    curr_score = score(iter - 1, :);
    curr_score = curr_score(curr_score ~= 0);
    next_score = score(iter + iter_increment - 1, :);
    next_score = next_score(next_score ~= 0);
    
    %% Plot
    if(strcmp(refinement_mode, 'cell_score') || ...
            strcmp(refinement_mode, 'cell_score_full'))
        curr_score = - curr_score;
    end
    if plt_iter
        imagesc(reshape(rf2fem*(curr_score'), numel(coarseGridX),...
            numel(coarseGridY))', 'Parent', sp1);
        sp1.GridLineStyle = 'none';
        sp1.XTick = [];
        sp1.YTick = [];
        sp1.Title.String = 'Elbo cell score';
        sp1.YDir = 'normal';
        gif
        drawnow
        if plot_split_only
            plt_iter = false;
        end
    end
    
    
    if numel(curr_score) ~= numel(next_score)
        [~, cell_index_pde] = max(curr_score)
        splt_cll = find(cell_dictionary == cell_index_pde)
        if splt_cll ~= modelParams.splitted_cells(curr_split)
            warning('splitted cell not identical with objective function')
            splt_cll = modelParams.splitted_cells(curr_split);
        end
        gridRF.split_cell(gridRF.cells{splt_cll});
        rf2fem = gridRF.map2fine(gridFEM);
        
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
        curr_split = curr_split + 1;
        plt_iter = true;
    end

end