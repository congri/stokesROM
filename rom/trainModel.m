%% This is the main training file for the Darcy-type ROM for Stokes equation
%% Preamble:

clear
addpath('./featureFunctions/nonOverlappingPolydisperseSpheres')
addpath('./mesh')
addpath('./aux')
addpath('./comp')
addpath('./FEM')
addpath('rom')
addpath('./VI')

%random number seed based on time
rng('shuffle');


%% Initialization
%Which data samples for training?
nTrain = 16;
% nStart = randi(1023 - nTrain); 
nStart = 1;
samples = nStart:(nTrain - 1 + nStart);
loadParams = false;     %load parameters from previous run?

rom = StokesROM;

rom.trainingData = StokesData(samples);
rom.trainingData.readData('px');
rom.trainingData.countVertices();

if loadParams
    disp('Loading modelParams...')
    load('./data/modelParams.mat');
    rom.modelParams = copy(modelParams);
    if any(rom.modelParams.interpolationMode)
        rom.trainingData.interpolate(rom.modelParams);
        rom.modelParams.fineScaleInterp(rom.trainingData.X_interp);
        interp = true;
    else
        rom.modelParams.fineScaleInterp(rom.trainingData.X);%for W_cf
        interp = false;
    end
    disp('... modelParams loaded.')
else
    rom.modelParams = ModelParams(rom.trainingData.u_bc, rom.trainingData.p_bc);
    rom.modelParams.initialize(nTrain);
    if any(rom.modelParams.interpolationMode)
        rom.trainingData.interpolate(rom.modelParams);
        rom.modelParams.fineScaleInterp(rom.trainingData.X_interp);
        interp = true;
    else
        rom.modelParams.fineScaleInterp(rom.trainingData.X);%for W_cf
        interp = false;
    end
end
%do not remove! if no cell is splitted, pass empty array
rom.modelParams.splitRFcells([]);

%Parameters from previous runs are deleted here
if exist('./data/', 'dir')
    rmdir('./data', 's'); %delete old params
end
mkdir('./data/');

rom.trainingData.shiftData(interp, 'p'); %shifts p to 0 at origin

rom.trainingData.vtx2Cell(rom.modelParams);

p_active_cells = {};
p_active_cells_S = {};
p_cell_score = {};

EMiter = 0;
nSplits = 3;
for split_iter = 1:(nSplits + 1)
    
    clear XMean XSqMean
    %delete design matrices so that they can be recomputed
    rom.trainingData.designMatrix = cell(1, nTrain);
    rom.trainingData.evaluateFeatures(rom.modelParams.gridRF);
    
    if strcmp(rom.modelParams.normalization, 'rescale')
        rom.trainingData.rescaleDesignMatrix;
    end
    
    if strcmp(rom.modelParams.mode, 'local')
        rom.trainingData.shapeToLocalDesignMat;
    end
    
    %theta_c must be initialized after design matrices exist
    if isempty(rom.modelParams.theta_c)
        disp('Initializing theta_c...')
        rom.modelParams.theta_c=0*ones(size(rom.trainingData.designMatrix{1},2), 1);
        disp('...done.')
    end
    
    %Step width for stochastic optimization in VI
    nRFc = rom.modelParams.gridRF.nCells;
    sw =[5e-3*ones(1, nRFc), 1e-4*ones(1, nRFc)];
    sw_decay = .98; %decay factor per iteration
    sw_min = 1e-1*sw;
    
    %% Actual training phase:
    
    converged = false;
    epoch = 0;  %one epoch == one time seen every data point
    ppool = parPoolInit(nTrain);
    pend = 0;
    EMiter_split = 0;
    while ~converged
        
        for n = 1:nTrain
            %Setting up a handle to the distribution q_n
            %this transfers less data in parfor loops
            P_n_minus_mu = rom.trainingData.P{n};
            if any(rom.modelParams.interpolationMode)
                W_cf_n = rom.modelParams.W_cf{1};
                S_n = rom.modelParams.sigma_cf.s0;
            else
                W_cf_n = rom.modelParams.W_cf{n};
                S_n = rom.modelParams.sigma_cf.s0(rom.trainingData.cellOfVertex{n});
            end
            %S_n is a vector of variances at vertices
            S_cf_n.sumLogS = sum(log(S_n));
            S_cf_n.Sinv_vec = (1./S_n)';%row vector
            Sinv = sparse(1:length(S_n), 1:length(S_n), S_cf_n.Sinv_vec);
            S_cf_n.WTSinv = (Sinv*W_cf_n)';
            
            tc.theta = rom.modelParams.theta_c;
            tc.Sigma = rom.modelParams.Sigma_c;
            tc.SigmaInv = inv(tc.Sigma);
            
            Phi_n = rom.trainingData.designMatrix{n};
            coarseMesh = rom.modelParams.coarseMesh;
            coarseMesh = coarseMesh.shrink;
            
            rf2fem = rom.modelParams.rf2fem;
            transType = rom.modelParams.diffTransform;
            transLimits = rom.modelParams.diffLimits;
            lg_q{n} = @(Xi) log_q_n(Xi, P_n_minus_mu, W_cf_n, S_cf_n, tc, Phi_n,...
                coarseMesh, transType, transLimits, rf2fem, true);
            lg_q_max{n} = @(Xi) log_q_n(Xi, P_n_minus_mu, W_cf_n, S_cf_n, tc,...
                Phi_n, coarseMesh, transType, transLimits, rf2fem, false);
        end
        varDistParamsVec = rom.modelParams.varDistParamsVec;
        
        if epoch > 0
            %Sequentially update N_threads qi's at a time, then perform M-step
            pstart = pend + 1;
            if pstart > nTrain
                pstart = 1;
                epoch = epoch + 1;
                %Gradually reduce VI step width
                sw = sw_decay*sw;
                sw(sw < sw_min) = sw_min(sw < sw_min);
            end
            pend = pstart + ppool.NumWorkers - 1;
            if pend > nTrain
                pend = nTrain;
            elseif pend < pstart
                pend = pstart;
            end
        else
            pstart = 1;
            pend = nTrain;
            epoch = epoch + 1;
        end
        
        disp('Variational Inference...')
        ticBytes(gcp);
        tic
        parfor n = pstart:pend
            %         mx{n} = max_fun(lg_q_max{n}, varDistParamsVec{n}(1:nRFc));
            %         varDistParamsVec{n}(1:nRFc) = mx{n};
            %Finding variational approximation to q_n
            [varDistParams{n}, varDistParamsVec{n}] =...
                efficientStochOpt(varDistParamsVec{n}, lg_q{n}, 'diagonalGauss',...
                sw, nRFc);
        end
        tocBytes(gcp)
        VI_time = toc
        rom.modelParams.varDistParamsVec = varDistParamsVec;
        disp('... VI done.')
        
        for n = pstart:pend
            %Compute expected values under variational approximation
            rom.modelParams.variational_mu{n} = varDistParams{n}.mu;
            rom.modelParams.variational_sigma{n} = varDistParams{n}.sigma;
            XMean(:, n) = varDistParams{n}.mu';
            XSqMean(:, n) = varDistParams{n}.XSqMean;
            
            P_n_minus_mu = rom.trainingData.P{n};
            if any(rom.modelParams.interpolationMode)
                W_cf_n = rom.modelParams.W_cf{1};
            else
                W_cf_n = rom.modelParams.W_cf{n};
            end
            p_cf_expHandle_n = @(X) sqMisfit(X, transType, transLimits,...
                coarseMesh, P_n_minus_mu, W_cf_n, rf2fem);
            %Expectations under variational distributions
            p_cf_exp =...
                mcInference(p_cf_expHandle_n, 'diagonalGauss', varDistParams{n});
            sqDist{n} = p_cf_exp;
        end
        
        
        %M-step: determine optimal parameters given the sample set
        tic
        disp('M-step...')
        rom.M_step(XMean, XSqMean, sqDist)
        disp('...M-step done.')
        M_step_time = toc
        
        rom.modelParams.compute_elbo(nTrain, XMean, XSqMean);
        elbo = rom.modelParams.elbo
        
        if ~mod(EMiter_split, 20)
            activeCells_S = rom.findMeshRefinement(true)'
            filename = './data/activeCells_S';
            save(filename, 'activeCells_S', '-ascii', '-append');
            activeCells = rom.findMeshRefinement(false)'
            filename = './data/activeCells';
            save(filename, 'activeCells', '-ascii', '-append');
        end
        
        
        
        Lambda_eff1_mode = conductivityBackTransform(...
            rom.trainingData.designMatrix{1}*rom.modelParams.theta_c, transType,...
            transLimits)
        rom.modelParams.printCurrentParams;
        plt = feature('ShowFigureWindows');
        if plt
            %plot current parameters
            if ~exist('figParams')
                figParams = figure('units','normalized','outerposition',[0 0 .5 1]);
            end
            rom.modelParams.plot_params(figParams, EMiter);
            
            % Plot data and reconstruction (modal value)
            if ~exist('figResponse')
                figResponse =...
                    figure('units','normalized','outerposition',[0 0 1 1]);
            end
            %plot modal lambda_c and corresponding -training- data reconstruction
            rom.plotCurrentState(figResponse, 0, transType, transLimits);
            
            if ~isempty(rom.modelParams.elbo)
                if ~exist('figElbo')
                    figElbo =...
                        figure('units','normalized','outerposition',[0 0 1 1]);
                end
                rom.modelParams.plotElbo(figElbo, EMiter);
            end
            
            % Plot data and reconstruction (modal value)
            if ~exist('figCellScore')
                figCellScore =...
                    figure('units','normalized','outerposition',[0 0 1 1]);
            end
            %plot cell scores
            sp1 = subplot(2, 3, 1, 'Parent', figCellScore);
            imagesc(reshape(rf2fem*(-rom.modelParams.cell_score),...
                numel(rom.modelParams.coarseGridX),...
                numel(rom.modelParams.coarseGridY))', 'Parent', sp1);
            sp1.YDir = 'normal';
            axis(sp1, 'square');
            sp1.GridLineStyle = 'none';
            sp1.XTick = [];
            sp1.YTick = [];
            cbp_lambda = colorbar('Parent', figCellScore);
            
            sp2 = subplot(2, 3, 2, 'Parent', figCellScore);
            imagesc(reshape(rf2fem*log(activeCells'),...
                numel(rom.modelParams.coarseGridX),...
                numel(rom.modelParams.coarseGridY))', 'Parent', sp2);
            sp2.YDir = 'normal';
            axis(sp2, 'square');
            p2.GridLineStyle = 'none';
            sp2.XTick = [];
            sp2.YTick = [];
            cbp_lambda = colorbar('Parent', figCellScore);
            
            sp3 = subplot(2, 3, 3, 'Parent', figCellScore);
            imagesc(reshape(rf2fem*log(activeCells_S'),...
                numel(rom.modelParams.coarseGridX),...
                numel(rom.modelParams.coarseGridY))', 'Parent', sp3);
            sp3.YDir = 'normal';
            axis(sp3, 'square');
            p2.GridLineStyle = 'none';
            sp3.XTick = [];
            sp3.YTick = [];
            cbp_lambda = colorbar('Parent', figCellScore);
            
            sp4 = subplot(2, 3, 4, 'Parent', figCellScore);
            map = colormap(sp4, 'lines');
            if(numel(rom.modelParams.cell_score) ~= numel(p_cell_score))
                for k = (numel(p_cell_score) + 1):...
                        numel(rom.modelParams.cell_score)
                    p_cell_score{k} = animatedline('Parent', sp4);
                    p_cell_score{k}.LineWidth = 2;
                    p_cell_score{k}.Marker = 'x';
                    p_cell_score{k}.MarkerSize = 10;
                    p_cell_score{k}.Color = map(k, :);
                end
                sp4.XLabel.String = 'iteration';
                sp4.YLabel.String = 'cell score';
            end
            for k = 1:numel(rom.modelParams.cell_score)
                addpoints(p_cell_score{k}, EMiter, -rom.modelParams.cell_score(k));
            end
            axis(sp4, 'tight');
            axis(sp4, 'fill');
            
            sp5 = subplot(2, 3, 5, 'Parent', figCellScore);
            if(numel(p_active_cells) ~= numel(activeCells))
                for k = (numel(p_active_cells) + 1):numel(activeCells)
                    p_active_cells{k} = animatedline('Parent', sp5);
                    p_active_cells{k}.LineWidth = 2;
                    p_active_cells{k}.Marker = 'x';
                    p_active_cells{k}.MarkerSize = 10;
                    p_active_cells{k}.Color = map(k, :);
                end
                sp5.XLabel.String = 'iteration';
                sp5.YLabel.String = 'active cell score';
            end
            for k = 1:numel(activeCells)
                addpoints(p_active_cells{k}, EMiter, log(activeCells(k)));
            end
            axis(sp5, 'tight');
            axis(sp5, 'fill');
            
            sp6 = subplot(2, 3, 6, 'Parent', figCellScore);
            if(numel(p_active_cells_S) ~= numel(activeCells_S))
                for k = (numel(p_active_cells_S) + 1):numel(activeCells_S)
                    p_active_cells_S{k} = animatedline('Parent', sp6);
                    p_active_cells_S{k}.LineWidth = 2;
                    p_active_cells_S{k}.Marker = 'x';
                    p_active_cells_S{k}.MarkerSize = 10;
                    p_active_cells_S{k}.Color = map(k, :);
                end
                sp6.XLabel.String = 'iteration';
                sp6.YLabel.String = 'active cell score with S';
            end
            for k = 1:numel(activeCells_S)
                addpoints(p_active_cells_S{k}, EMiter, log(activeCells_S(k)));
            end
            axis(sp6, 'tight');
            axis(sp6, 'fill');
            drawnow;
        end
        
        %write parameters to disk to investigate convergence
        rom.modelParams.write2file('thetaPriorHyperparam');
        rom.modelParams.write2file('theta_c');
        rom.modelParams.write2file('sigma_c');
        rom.modelParams.write2file('elbo');
        rom.modelParams.write2file('cell_score');
        if ~mod(EMiter, 10)
            tic
            modelParams = copy(rom.modelParams);
            modelParams.p_theta = [];
            modelParams.p_sigma = [];
            modelParams.p_gamma = [];
            modelParams.p_elbo = [];
            %Save modelParams after every iteration
            disp('Saving modelParams...')
            save('./data/modelParams.mat', 'modelParams', '-v7.3');
            disp('...modelParams saved.')
            save_time = toc
        end
        save('./data/XMean', 'XMean');
        save('./data/XSqMean', 'XSqMean');
        
        if epoch > rom.modelParams.max_EM_iter
            converged = true;
        else
            EMiter = EMiter + 1
            EMiter_split = EMiter_split + 1
            epoch
        end
    end
    
    if split_iter < (nSplits + 1)
        disp('splitting cell...')
        activeCells_S = rom.findMeshRefinement(true)'
        filename = './data/activeCells_S';
        save(filename, 'activeCells_S', '-ascii', '-append');
        
        [~, cell_index_pde] = max(activeCells_S);
        cell_index = find(rom.modelParams.cell_dictionary == cell_index_pde)
        rom.modelParams.splitRFcells([cell_index]);
        disp('...cell splitted.')
    end
end

predictionScript;
