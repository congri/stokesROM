%% This is the main training file for the Darcy-type ROM for Stokes equation
%% Preamble:

clear
addpath('./featureFunctions/nonOverlappingPolydisperseSpheres')
addpath('./mesh')
%% Define parameters here:

samples = 0:15;
max_EM_iter = 800;  %maximum EM iterations
muField = 0;        %mean function in p_cf

mode = 'local';
normalization = 'rescale';

%Conductivity transformation options
condTransOpts.type = 'log';
condTransOpts.limits = [1e-16, 1e16];

%grid vectors
gridX = ones(1, 4);   %coarse FEM
gridX = gridX/sum(gridX);
gridY = gridX;
gridRF = RectangularMesh([.25 .25 .25 .25]);
gridSX = ones(1, 128);   %p_cf S grid
gridSX = gridSX/sum(gridSX);
gridSY = gridSX;

%Boundary condition fields
p_bc = @(x) 0;
%influx?
% u_bc{1} = 'u_x=0.25 - (x[1] - 0.5)*(x[1] - 0.5)';
% u_bc{2} = 'u_y=0.0';
u_bc{1} = 'u_x=-0.8 + 2.0*x[1]';
u_bc{2} = 'u_y=-1.2 + 2.0*x[0]';
%% Initialize reduced order model object:
rom = StokesROM;


%% Read training data, initialize parameters and evaluate features:

rom.readTrainingData(samples, u_bc);
N_train = numel(rom.trainingData.samples);
rom.trainingData.countVertices();

rom.modelParams = ModelParams;
rom.modelParams.interpolationMode = 'cubic';  %Interpolation on regular 
                                              %finescale grid, including solid
                                              %phase
rom.modelParams.smoothingParameter = [];      %only applies for interp. data;
                                              %empty for no smoothing
rom.initializeModelParams(p_bc, u_bc, '', gridX, gridY, gridRF, gridSX, gridSY);

rom.modelParams.condTransOpts = condTransOpts;
if any(rom.modelParams.interpolationMode)
    rom.trainingData.interpolate(gridSX, gridSY,...
        rom.modelParams.interpolationMode, rom.modelParams.smoothingParameter);
    rom.modelParams.fineScaleInterp(rom.trainingData.X_interp);
else
    rom.modelParams.fineScaleInterp(rom.trainingData.X);%for W_cf
end
rom.modelParams.saveParams('gtcscscf');
rom.modelParams.saveParams('coarseMesh');
rom.modelParams.saveParams('priorType');
rom.modelParams.saveParams('condTransOpts');
rom.modelParams.saveParams('gridRF');
rom.modelParams.saveParams('gridS');
rom.modelParams.saveParams('interp');
rom.modelParams.saveParams('smooth');
rom.trainingData.evaluateFeatures(gridRF);

if strcmp(normalization, 'rescale')
    rom.trainingData.rescaleDesignMatrix;
end

if strcmp(mode, 'local')
    rom.trainingData.shapeToLocalDesignMat;
end
%theta_c must be initialized after design matrices exist
rom.modelParams.theta_c = 0*ones(size(rom.trainingData.designMatrix{1}, 2), 1);
rom.trainingData.vtxToCell(gridSX, gridSY, rom.modelParams.interpolationMode);

%Step width for stochastic optimization in VI
sw =[4e-2*ones(1, gridRF.nCells), 1e-3*ones(1, gridRF.nCells)];

%% Bring variational distribution params in form for unconstrained optimization
varDistParamsVec{1} = [rom.modelParams.variational_mu{1},...
    -2*log(rom.modelParams.variational_sigma{1})];
varDistParamsVec = repmat(varDistParamsVec, N_train, 1);
%% 
%% Actual training phase:
% 

converged = false;
EMiter = 0;
epoch = 0;  %one epoch == one time seen every data point
thetaArray = [];
SigmaArray = [];
ppool = parPoolInit(N_train);
while ~converged
    
    for n = 1:N_train
        %Setting up a handle to the distribution q_n
        %this transfers less data in parfor loops
        P_n_minus_mu = rom.trainingData.P{n} - muField;
        if any(rom.modelParams.interpolationMode)
            W_cf_n = rom.modelParams.W_cf{1};
            S_n = rom.modelParams.sigma_cf.s0;
        else
            W_cf_n = rom.modelParams.W_cf{n};
            S_n = rom.modelParams.sigma_cf.s0(rom.trainingData.cellOfVertex{n});
        end
        %S_n is a vector of variances at vertices
        S_cf_n.sumLogS = sum(log(S_n));
        S_cf_n.Sinv_vec = 1./S_n;
        Sinv = sparse(1:length(S_n), 1:length(S_n), S_cf_n.Sinv_vec);
        S_cf_n.WTSinv = (Sinv*W_cf_n)';
        
        tc.theta = rom.modelParams.theta_c;
        tc.Sigma = rom.modelParams.Sigma_c;
        tc.SigmaInv = inv(tc.Sigma);
        
        Phi_n = rom.trainingData.designMatrix{n};
        coarseMesh = rom.modelParams.coarseMesh;
        
        rf2fem = rom.modelParams.rf2fem;
        lg_q{n} = @(Xi) log_q_n(Xi, P_n_minus_mu, W_cf_n, S_cf_n, tc, Phi_n,...
            coarseMesh, condTransOpts, rf2fem);
    end
    
    nRFc = gridRF.nCells;
    tic
    parfor n = 1:N_train
        %Finding variational approximation to q_n
        [varDistParams{n}, varDistParamsVec{n}] =...
            efficientStochOpt(varDistParamsVec{n}, lg_q{n}, 'diagonalGauss',...
            sw, nRFc);
    end
    VI_time = toc
    
    for n = 1:N_train
        %Compute expected values under variational approximation
        XMean(:, n) = varDistParams{n}.mu';
        XSqMean(:, n) = varDistParams{n}.XSqMean;
        
        P_n_minus_mu = rom.trainingData.P{n} - muField;
        if any(rom.modelParams.interpolationMode)
            W_cf_n = rom.modelParams.W_cf{1};
        else
            W_cf_n = rom.modelParams.W_cf{n};
        end
        p_cf_expHandle_n = @(X) sqMisfit(X, condTransOpts,...
            coarseMesh, P_n_minus_mu, W_cf_n, rf2fem);
        %Expectations under variational distributions
        p_cf_exp =...
            mcInference(p_cf_expHandle_n, 'diagonalGauss', varDistParams{n});
        sqDist{n} = p_cf_exp;
    end
    
    
    % romObj.varExpect_p_cf_exp_mean = mean(tempArray, 2);
    
    %M-step: determine optimal parameters given the sample set
    tic
    rom.M_step(XMean, XSqMean, sqDist);
    M_step_time = toc
    
    
    
    Lambda_eff1_mode = conductivityBackTransform(...
        rom.trainingData.designMatrix{1}*rom.modelParams.theta_c, condTransOpts)
    rom.modelParams.printCurrentParams(mode);
    plt = feature('ShowFigureWindows');
    if plt
        %plot current parameters
        if ~exist('figParams')
            figParams = figure;
        end
        thetaArray = [thetaArray, rom.modelParams.theta_c];
        SigmaArray = [SigmaArray, full(diag(rom.modelParams.Sigma_c))];
        rom.modelParams.plot_params(figParams, thetaArray', SigmaArray');
                
        % Plot data and reconstruction (modal value)
        if ~exist('figResponse')
            figResponse = figure;
        end
        %plot modal lambda_c and corresponding -training- data reconstruction
        rom.plotCurrentState(figResponse, 3, condTransOpts);
    end
    
    %collect data and write it to disk periodically to save memory
    rom.modelParams.saveParams('gtcscscfvardist') 
    rom.modelParams.saveParams('stc')
    save('./data/XMean', 'XMean');
    save('./data/XSqMean', 'XSqMean');
    
    if EMiter > max_EM_iter
        converged = true;
    else
        EMiter = EMiter + 1
    end
end