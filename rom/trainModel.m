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
nStart = randi(1023 - nTrain);     samples = nStart:(nTrain - 1 + nStart);
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

%Parameters from previous runs are deleted here
if exist('./data/', 'dir')
    rmdir('./data', 's'); %delete old params
end
mkdir('./data/');

rom.trainingData.shiftData(interp, 'p'); %shifts p to 0 at origin
rom.trainingData.evaluateFeatures(rom.modelParams.gridRF);

if strcmp(rom.modelParams.normalization, 'rescale')
    rom.trainingData.rescaleDesignMatrix;
end

if strcmp(rom.modelParams.mode, 'local')
    rom.trainingData.shapeToLocalDesignMat;
end
%theta_c must be initialized after design matrices exist
rom.modelParams.theta_c = 0*ones(size(rom.trainingData.designMatrix{1}, 2), 1);

rom.trainingData.vtx2Cell(rom.modelParams);


%Step width for stochastic optimization in VI
nRFc = rom.modelParams.gridRF.nCells;
sw =[3e-2*ones(1, nRFc), 1e-2*ones(1, nRFc)];
sw_decay = .98; %decay factor per iteration
sw_min = 1e-3*sw;

%% Actual training phase:

converged = false;
EMiter = 0;
epoch = 0;  %one epoch == one time seen every data point
ppool = parPoolInit(nTrain);
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
    
    disp('Variational Inference...')
    ticBytes(gcp);
    tic
    parfor n = 1:nTrain
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
    
    %Gradually reduce VI step width
    sw = sw_decay*sw;
    sw(sw < sw_min) = sw_min(sw < sw_min);
    
    for n = 1:nTrain
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
    rom.M_step(XMean, XSqMean, sqDist)
    M_step_time = toc
    
    rom.modelParams.compute_elbo(nTrain, XMean, XSqMean);
    
    
    
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
        %Save modelParams after every iteration
        disp('Saving modelParams...')
        save('./data/modelParams.mat', 'modelParams', '-v7.3');
        disp('...modelParams saved.')
        save_time = toc
    end
    save('./data/XMean', 'XMean');
    save('./data/XSqMean', 'XSqMean');
    
    if EMiter > rom.modelParams.max_EM_iter
        converged = true;
    else
        EMiter = EMiter + 1
    end
end

predictionScript;
