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
nStart = 0;
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
%     rom.modelParams = ModelParams(rom.trainingData.u_bc, '10.0');
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

sw0_mu = 1e-3;
sw0_sigma = 1e-6;
sw_decay = .995; %decay factor per iteration
nSplits = 20;
tic_tot = tic;
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
    sw = (sw_decay^rom.modelParams.EM_iter)*...
        [sw0_mu*ones(1, nRFc), sw0_sigma*ones(1, nRFc)];
    sw_min = 3e-1*[sw0_mu*ones(1, nRFc), sw0_sigma*ones(1, nRFc)];
    sw(sw < sw_min) = sw_min(sw < sw_min);
    
    %% Actual training phase:
    
    converged = false;
    epoch = 0;  %one epoch == one time seen every data point
    ppool = parPoolInit(nTrain);
    pend = 0;
    rom.modelParams.EM_iter_split = 0;
    while ~converged
        
        %% Setting up a handle to the distribution q_n - this transfers less 
        %data in parfor loops
        for n = 1:nTrain
            P_n_minus_mu = rom.trainingData.P{n};
            if any(rom.modelParams.interpolationMode)
                W_cf_n = rom.modelParams.W_cf{1};
                S_n = rom.modelParams.sigma_cf.s0;
            else
                W_cf_n = rom.modelParams.W_cf{n};
                S_n = rom.modelParams.sigma_cf.s0(...
                    rom.trainingData.cellOfVertex{n});
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
            lg_q{n} = @(Xi) log_q_n(Xi, P_n_minus_mu, W_cf_n, S_cf_n, tc,...
                Phi_n, coarseMesh, transType, transLimits, rf2fem, true);
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
            [varDistParams{n}, varDistParamsVec{n}] = efficientStochOpt(...
                varDistParamsVec{n}, lg_q{n}, 'diagonalGauss', sw, nRFc);
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
                mcInference(p_cf_expHandle_n,'diagonalGauss', varDistParams{n});
            sqDist{n} = p_cf_exp;
        end
        
        
        %M-step: determine optimal parameters given the sample set
        tic
        disp('M-step...')
        rom.M_step(XMean, XSqMean, sqDist)
        disp('...M-step done.')
        M_step_time = toc
        
        if ~exist('figElboTest')
            figElboTest =...
                figure('units','normalized','outerposition',[0 0 1 1]);
        end
        rom.modelParams.compute_elbo(nTrain, XMean, XSqMean,...
            rom.trainingData.X_interp{1}, figElboTest);
        elbo = rom.modelParams.elbo
        
%         if ~mod(rom.modelParams.EM_iter_split - 1, 20)
        if false
            rom.modelParams.active_cells_S = rom.findMeshRefinement(true)';
            activeCells_S = rom.modelParams.active_cells_S
            filename = './data/activeCells_S';
            save(filename, 'activeCells_S', '-ascii', '-append');
            rom.modelParams.active_cells = rom.findMeshRefinement(false)';
            activeCells = rom.modelParams.active_cells
            filename = './data/activeCells';
            save(filename, 'activeCells', '-ascii', '-append');
        end
        
        %Print some output
        Lambda_eff1_mode = conductivityBackTransform(...
            rom.trainingData.designMatrix{1}*rom.modelParams.theta_c,...
            transType, transLimits)
        rom.modelParams.printCurrentParams;
        
        disp('Plotting...')
        t_plt = tic;
        %plot parameters
        rom.modelParams.plot_params();
        %plot modal lambda_c and corresponding -training- data reconstruction
        rom.plotCurrentState(0, transType, transLimits);
        %plot elbo vs. training iteration
        t_tot = toc(tic_tot)
        rom.modelParams.plotElbo(t_tot);
        %Plot adaptive refinement cell scores
        rom.modelParams.plotCellScores();
        disp('...plotting done. Plotting time:')
        t_plt = toc(t_plt)
        
        
        %write parameters to disk to investigate convergence
        rom.modelParams.write2file('thetaPriorHyperparam');
        rom.modelParams.write2file('theta_c');
        rom.modelParams.write2file('sigma_c');
        rom.modelParams.write2file('elbo');
        rom.modelParams.write2file('cell_score');
        rom.modelParams.write2file('cell_score_full');
        if ~mod(rom.modelParams.EM_iter, 10)
            tic
            modelParams = copy(rom.modelParams);
            modelParams.pParams = [];
            modelParams.pElbo = [];
            modelParams.pCellScores = [];
            %Save modelParams after every iteration
            disp('Saving modelParams...')
            save('./data/modelParams.mat', 'modelParams', '-v7.3');
            disp('...modelParams saved.')
            save_time = toc
        end
        save('./data/XMean', 'XMean');
        save('./data/XSqMean', 'XSqMean');
        
        if epoch > rom.modelParams.max_EM_epochs
            converged = true;
        end
        epoch
    end
    
    if split_iter < (nSplits + 1)
        disp('splitting cell...')
        refinement_objective = 'full_elbo_score';
        if strcmp(refinement_objective, 'active_cells_S')
            rom.modelParams.active_cells_S = rom.findMeshRefinement(true)';
            activeCells_S = rom.modelParams.active_cells_S;
            filename = './data/activeCells_S';
            save(filename, 'activeCells_S', '-ascii', '-append');
            [~, cell_index_pde] = max(rom.modelParams.active_cells_S);
        elseif strcmp(refinement_objective, 'active_cells')
            rom.modelParams.active_cells = rom.findMeshRefinement(false)';
            activeCells = rom.modelParams.active_cells;
            filename = './data/activeCells';
            save(filename, 'activeCells', '-ascii', '-append');
            [~, cell_index_pde] = max(rom.modelParams.active_cells);
        elseif strcmp(refinement_objective, 'full_elbo_score')
            [~, cell_index_pde] = min(rom.modelParams.cell_score_full);
        elseif strcmp(refinement_objective, 'reduced_elbo_score')
            [~, cell_index_pde] = min(rom.modelParams.cell_score);
        elseif strcmp(refinement_objective, 'random')
            cell_index_pde = randi(numel(rom.modelParams.cell_score));
        end
        
        cell_index = find(rom.modelParams.cell_dictionary == cell_index_pde)
        rom.modelParams.splitRFcells([cell_index]);
        disp('...cell splitted.')
    end
end

predictionScript;
