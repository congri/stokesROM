classdef StokesROM < handle
    %Class for reduced order model of Stokes equation
    
    properties
        
        %StokesData object
        trainingData
        
        %Model parameters
        modelParams
        
    end
    
    methods
        function [self] = StokesROM()
            %Constructor
        end
        
        function readTrainingData(self, samples, u_bc, p_bc)
            %Sets trainingData to StokesData object
            self.trainingData = StokesData(samples, u_bc, p_bc);
            self.trainingData.readData('px');
        end
        
        function initializeModelParams(self, p_bc, u_bc, mode, gridX, gridY,...
                gridRF, gridSX, gridSY)
            %Initialize params theta_c, theta_cf
            
            self.modelParams.gridSX = gridSX;
            self.modelParams.gridSY = gridSY;
            self.modelParams.gridRF = gridRF;
            self.modelParams.rf2fem = gridRF.map2fine(gridX, gridY);
            
            %Coarse mesh object
            self.modelParams.coarseMesh = MeshFEM(gridX, gridY);
            nX = length(gridX); nY = length(gridY);
            
            %Convert flow bc string to handle functions
            u_x_temp = strrep(u_bc{1}(5:end), 'x[1]', 'y');
            %This is only valid for unit square domain!!
            u_x_temp_le = strrep(u_x_temp, 'x[0]', '0');
            u_x_temp_r = strrep(u_x_temp, 'x[0]', '1');
            
            u_y_temp = strrep(u_bc{2}(5:end), 'x[0]', 'x');
            u_y_temp_lo = strrep(u_y_temp, 'x[1]', '0');
            u_y_temp_u = strrep(u_y_temp, 'x[1]', '1');
            u_bc_handle{1} = str2func(strcat('@(x)', '-(', u_y_temp_lo, ')'));
            u_bc_handle{2} = str2func(strcat('@(y)', u_x_temp_r));
            u_bc_handle{3} = str2func(strcat('@(x)', u_y_temp_u));
            u_bc_handle{4} = str2func(strcat('@(y)', '-(', u_x_temp_le, ')'));
            
            self.modelParams.coarseMesh =...
                self.modelParams.coarseMesh.setBoundaries(2:(2*nX + 2*nY),...
                p_bc, u_bc_handle);
            
            nData = numel(self.trainingData.samples);
            
            self.modelParams.initialize(gridRF.nCells, nData, mode);
                        
            %Parameters from previous runs can be deleted here
            if exist('./data/', 'dir')
                rmdir('./data', 's'); %delete old params
            end
        end
        
        function M_step(self, XMean, XSqMean, sqDist_p_cf)
            
            if(strcmp(self.modelParams.prior_theta_c, 'VRVM') || ...
                    strcmp(self.modelParams.prior_theta_c, 'sharedVRVM'))
                dim_theta = numel(self.modelParams.theta_c);
                nTrain = numel(self.trainingData.samples);
                nElc = size(self.trainingData.designMatrix{1}, 1);
                
                %Parameters that do not change when q(lambda_c) is fixed
                a = self.modelParams.VRVM_a + .5;
                e = self.modelParams.VRVM_e + .5*nTrain;
                c = self.modelParams.VRVM_c + .5*nTrain;
                Ncells_gridS = numel(self.modelParams.gridSX)*...
                    numel(self.modelParams.gridSY);
                if any(self.modelParams.interpolationMode)
                    sqDistSum = 0;
                    for n = 1:numel(self.trainingData.samples)
                        sqDistSum = sqDistSum + sqDist_p_cf{n};
                    end
                else
                    sqDistSum = zeros(Ncells_gridS, 1);
                    for j = 1:Ncells_gridS
                        for n = 1:numel(self.trainingData.samples)
                            sqDistSum(j)= sqDistSum(j) + mean(sqDist_p_cf{n}(...
                                j == self.trainingData.cellOfVertex{n}));
                        end
                    end
                end
                f = self.modelParams.VRVM_f + .5*sqDistSum;
                tau_cf = e./f;  %p_cf precision
                
                %initialization
                if(numel(self.modelParams.gamma) ~= dim_theta)
                    warning('resizing theta precision parameter gamma')
                    self.modelParams.gamma = 1e0*ones(dim_theta, 1);
                end
                gamma = self.modelParams.gamma;
                tau_theta = diag(gamma);    %precision of q(theta_c)
                if isempty(self.modelParams.Sigma_theta_c)
                    Sigma_theta = inv(tau_theta);
                else
                    Sigma_theta = self.modelParams.Sigma_theta_c;
                end
                mu_theta = self.modelParams.theta_c;
                
                for i = 1:self.modelParams.VRVM_iter
                    b = self.modelParams.VRVM_b + .5*(mu_theta.^2 +...
                            diag(Sigma_theta));
                    if strcmp(self.modelParams.prior_theta_c, 'sharedVRVM')
                        b = reshape(b, dim_theta/nElc, nElc);
                        b = mean(b, 2);
                        b = repmat(b, nElc, 1);
                    end
                    gamma = a./b;
                    d = self.modelParams.VRVM_d + .5*sum(XSqMean, 2);
                    for n = 1:nTrain
                        PhiThetaMean_n = self.trainingData.designMatrix{n}*...
                            mu_theta;
                        d = d - XMean(:, n).*PhiThetaMean_n;
                        PhiThetaSq_n = diag(PhiThetaMean_n*PhiThetaMean_n'+...
                            self.trainingData.designMatrix{n}*...
                            Sigma_theta*self.trainingData.designMatrix{n}');
                        d = d + .5*PhiThetaSq_n;
                    end
                    tau_c = c./d;   %precision of p_c
                    sqrt_tau_c = sqrt(tau_c);
                    tau_theta = diag(gamma);
                    sumPhiTau_cXMean = 0;
                    for n = 1:nTrain
                        %to ensure pos. def.
                        A = diag(sqrt_tau_c)*self.trainingData.designMatrix{n};
                        tau_theta = tau_theta + A'*A;
%                         tau_theta = tau_theta +...
%                             self.trainingData.designMatrix{n}'*diag(tau_c)*...
%                             self.trainingData.designMatrix{n};
                        sumPhiTau_cXMean = sumPhiTau_cXMean + ...
                            self.trainingData.designMatrix{n}'*...
                            diag(tau_c)*XMean(:, n);
                    end
                    Sigma_theta = inv(tau_theta);
                    mu_theta = Sigma_theta*sumPhiTau_cXMean;
                end
                
                %assign <S>, <Sigma_c>, <theta_c>
                self.modelParams.sigma_cf.s0 = 1./tau_cf;
                
                self.modelParams.Sigma_c = diag(1./tau_c);
                self.modelParams.theta_c = mu_theta;
                self.modelParams.Sigma_theta_c = Sigma_theta;
                
                self.modelParams.gamma = gamma;
                mean_s0 = mean(self.modelParams.sigma_cf.s0)
            else
                %Update model parameters
                self = self.update_p_c(XMean, XSqMean);
                self = self.update_p_cf(sqDist_p_cf);
            end
        end
        
        function update_p_c(self, XMean, XSqMean)
            %% Find optimal theta_c and Sigma_c self-consistently:
            %update theta_c, then Sigma_c, then theta_c and so on
            
            %short-hand notation
            dim_theta_c = numel(self.modelParams.theta_c);
            N_train = numel(self.trainingData.samples);
            nElc = self.modelParams.coarseMesh.nEl;
            nFeatures = dim_theta_c/nElc; %for shared RVM only!
            
            %Start from previous best estimate
            curr_theta_c = self.modelParams.theta_c;
            I = speye(dim_theta_c);
            Sigma_c = self.modelParams.Sigma_c;
            
            %sum_i Phi_i^T Sigma^-1 <X^i>_qi
            sumPhiTSigmaInvXmean = 0;
            sumPhiTSigmaInvXmeanOriginal = 0;
            Sigma_c_inv = inv(Sigma_c);
            Sigma_cInvXMean = Sigma_c\XMean;
            sumPhiTSigmaInvPhi = 0;
            sumPhiTSigmaInvPhiOriginal = 0;
            
            
            
            PhiThetaMat = zeros(nElc, N_train);
            
            for n = 1:N_train
                sumPhiTSigmaInvXmean = sumPhiTSigmaInvXmean +...
                    self.trainingData.designMatrix{n}'*Sigma_cInvXMean(:, n);
                sumPhiTSigmaInvPhi = sumPhiTSigmaInvPhi +...
                    self.trainingData.designMatrix{n}'*Sigma_c_inv*...
                    self.trainingData.designMatrix{n};
                PhiThetaMat(:, n) = self.trainingData.designMatrix{n}*...
                    curr_theta_c;
            end
            
            stabilityParam = 1e-2;    %for stability in matrix inversion
            if(strcmp(self.modelParams.prior_theta_c, 'adaptiveGaussian') ||...
                    strcmp(self.modelParams.prior_theta_c, 'RVM') || ...
                    strcmp(self.modelParams.prior_theta_c, 'sharedRVM'))
                %Find prior hyperparameter by max marginal likelihood
                
                if strcmp(self.modelParams.prior_theta_c, 'adaptiveGaussian')
                    self.modelParams.gamma = 1;
                elseif(strcmp(self.modelParams.prior_theta_c, 'RVM') ||...
                        strcmp(self.modelParams.prior_theta_c, 'sharedRVM'))
                    if(numel(self.modelParams.gamma) ~= dim_theta_c)
                        warning('resizing theta hyperparam')
                        self.modelParams.gamma = 1e-4*ones(dim_theta_c, 1);
                    end
                end
                converged = false;
                iter = 0;
                while(~converged)
                    if strcmp(self.modelParams.prior_theta_c,...
                            'adaptiveGaussian')
                        SigmaTilde = inv(sumPhiTSigmaInvPhi +...
                            (self.modelParams.gamma + stabilityParam)*I);
                        muTilde = SigmaTilde*sumPhiTSigmaInvXmean;
                        theta_prior_hyperparam_old = self.modelParams.gamma;
                        self.modelParams.gamma =...
                            dim_theta_c/(muTilde'*muTilde + trace(SigmaTilde));
                        
                    elseif(strcmp(self.modelParams.prior_theta_c, 'RVM') ||...
                            strcmp(self.modelParams.prior_theta_c, 'sharedRVM'))
                        SigmaTilde = inv(sumPhiTSigmaInvPhi +...
                            diag(self.modelParams.gamma));
                        muTilde = (sumPhiTSigmaInvPhi +...
                            diag(self.modelParams.gamma))\sumPhiTSigmaInvXmean;
                        theta_prior_hyperparam_old = self.modelParams.gamma;
                        
                        if strcmp(self.modelParams.prior_theta_c, 'RVM')
                            self.modelParams.gamma =...
                                1./(muTilde.^2 + diag(SigmaTilde));
                            
                        elseif strcmp(self.modelParams.prior_theta_c,...
                                'sharedRVM')
                            muTildeSq = muTilde.^2;
                            varTilde = diag(SigmaTilde);
                            lambdaInv = (1/nElc)*...
                                (sum(reshape(muTildeSq, nFeatures, nElc),2) +...
                                sum(reshape(varTilde, nFeatures, nElc), 2));
                            self.modelParams.gamma =...
                                repmat(1./lambdaInv, nElc, 1);
                        end
                        self.modelParams.gamma =...
                            self.modelParams.gamma + stabilityParam;
                    end
                    crit = norm(1./self.modelParams.gamma -...
                        1./theta_prior_hyperparam_old)/...
                        norm(1./self.modelParams.gamma);
                    if(crit < 1e-5 || iter >= 10)
                        converged = true;
                    elseif(any(~isfinite(self.modelParams.gamma)) ||...
                            any(self.modelParams.gamma <= 0))
                        converged = true;
                        muTilde.^2
                        self.modelParams.gamma
                        self.modelParams.gamma = ones(dim_theta_c, 1);
                        warning(strcat('Gaussian hyperparameter precision',...
                            'is negative or not a number. Setting it to 1.'))
                    end
                    iter = iter + 1;
                end
            end
            
            linsolveOpts.SYM = true;
            linsolveOpts.POSDEF = true;
            iter = 0;
            converged = false;
            while(~converged)
                theta_old = curr_theta_c;  %to check for iterative convergence
                
                %Matrix M is pos. def., invertible even if badly conditioned
                if strcmp(self.modelParams.prior_theta_c,'hierarchical_laplace')
                    offset = 1e-30;
                    U = diag(sqrt((abs(curr_theta_c) + offset)/...
                        self.modelParams.gamma(1)));
                elseif strcmp(self.modelParams.prior_theta_c,...
                        'hierarchical_gamma')
                    U = diag(sqrt((.5*abs(curr_theta_c).^2 +...
                        self.modelParams.gamma(2))./...
                        (self.modelParams.gamma(1) + .5)));
                elseif(strcmp(self.modelParams.prior_theta_c, 'gaussian') ||...
                        strcmp(self.modelParams.prior_theta_c,...
                        'adaptiveGaussian'))
                    sumPhiTSigmaInvPhi = sumPhiTSigmaInvPhi +...
                        (self.modelParams.gamma(1) + stabilityParam)*I;
                elseif(strcmp(self.modelParams.prior_theta_c, 'RVM') ||...
                        strcmp(self.modelParams.prior_theta_c, 'sharedRVM'))
                    sumPhiTSigmaInvPhi = sumPhiTSigmaInvPhi +...
                        diag(self.modelParams.gamma);
                elseif strcmp(self.modelParams.prior_theta_c, 'none')
                else
                    error('Unknown prior on theta_c')
                end
                
                if (strcmp(self.modelParams.prior_theta_c, 'gaussian') ||...
                        strcmp(self.modelParams.prior_theta_c, 'RVM') ||...
                        strcmp(self.modelParams.prior_theta_c, 'none') ||...
                        strcmp(self.modelParams.prior_theta_c, 'sharedRVM')||...
                        strcmp(self.modelParams.prior_theta_c,...
                        'adaptiveGaussian'))
                    theta_temp = sumPhiTSigmaInvPhi\sumPhiTSigmaInvXmean;
                    converged = true;   %is this true? We do not need to
                    %iteratively maximize theta
                else
                    theta_temp = U*linsolve((U*sumPhiTSigmaInvPhi*U + I), U,...
                        linsolveOpts)*sumPhiTSigmaInvXmean;
                end
                [~, msgid] = lastwarn;     %to catch nearly singular matrix
                
                if(strcmp(msgid, 'MATLAB:singularMatrix') ||...
                        strcmp(msgid, 'MATLAB:nearlySingularMatrix') ||...
                        strcmp(msgid, 'MATLAB:illConditionedMatrix') ||...
                        norm(theta_temp)/length(curr_theta_c) > 1e8)
                    curr_theta_c = .5*(curr_theta_c + .1*(norm(curr_theta_c)/...
                        norm(theta_temp))*theta_temp)
                    warning('theta_c is assumes strange values. Go tiny step.')
                    if any(~isfinite(curr_theta_c))
                        %restart from 0
                        warning(strcat('Some components of theta are not',...
                            'finite. Restarting from theta = 0...'))
                        curr_theta_c = 0*curr_theta_c;
                    end
                else
                    curr_theta_c = theta_temp;
                end
                
                PhiThetaMat = zeros(nElc, N_train);
                for n = 1:N_train
                    PhiThetaMat(:, n) =...
                        self.trainingData.designMatrix{n}*curr_theta_c;
                end
                
                
                Sigma_c = sparse(1:nElc, 1:nElc,...
                    mean(XSqMean - 2*(PhiThetaMat.*XMean) + PhiThetaMat.^2, 2));
                %Variances must be positive
                Sigma_c(logical(eye(size(Sigma_c)))) =...
                    abs(Sigma_c(logical(eye(size(Sigma_c)))));
                
                
                %% Recompute changed quantities
                Sigma_c_inv = inv(Sigma_c);
                Sigma_cInvXMean = Sigma_c\XMean;
                
                %sum_i Phi_i^T Sigma^-1 <X^i>_qi
                sumPhiTSigmaInvXmean = 0;
                sumPhiTSigmaInvPhi = 0;
                for n = 1:N_train
                    sumPhiTSigmaInvXmean = sumPhiTSigmaInvXmean +...
                        self.trainingData.designMatrix{n}'*Sigma_cInvXMean(:,n);
                    sumPhiTSigmaInvPhi = sumPhiTSigmaInvPhi +...
                        self.trainingData.designMatrix{n}'*...
                        (Sigma_c\self.trainingData.designMatrix{n});
                end
                
                iter = iter + 1;
                thetaDiffRel = norm(theta_old - curr_theta_c)/...
                    (norm(curr_theta_c)*numel(curr_theta_c));
                if((iter > 3 && thetaDiffRel < 1e-8) || iter > 200)
                    converged = true;
                end
            end
            
            %Assign final values
            self.modelParams.theta_c = curr_theta_c;
            self.modelParams.Sigma_c = Sigma_c;
        end
        
        function update_p_cf(self, sqDist_p_cf)
            
            Ncells_gridS = numel(self.modelParams.gridSX)*...
                numel(self.modelParams.gridSY);
            self.modelParams.sigma_cf.s0 = zeros(Ncells_gridS, 1);
            for j = 1:Ncells_gridS
                for n = 1:numel(self.trainingData.samples)
                    self.modelParams.sigma_cf.s0(j) =...
                        (1/n)*((n - 1)*self.modelParams.sigma_cf.s0(j) + ...
                        mean(sqDist_p_cf{n}(j ==...
                        self.trainingData.cellOfVertex{n})));
                end
            end
            mean_s0 = mean(self.modelParams.sigma_cf.s0)
        end
        
        function plotCurrentState(self, fig, dataOffset, condTransOpts)
            %Plots the current modal effective property and the modal
            %reconstruction for 2 -training- samples
            for i = 1:4
                Lambda_eff_mode = conductivityBackTransform(...
                    self.trainingData.designMatrix{i + dataOffset}*...
                    self.modelParams.theta_c, condTransOpts);
                Lambda_eff_mode = self.modelParams.rf2fem*...
                    Lambda_eff_mode;
                sb1 = subplot(4, 3, 1 + (i - 1)*3, 'Parent', fig);
                imagesc(reshape(Lambda_eff_mode,...
                    self.modelParams.coarseMesh.nElX,...
                    self.modelParams.coarseMesh.nElY)', 'Parent', sb1)
                sb1.YDir = 'normal';
                axis(sb1, 'tight');
                axis(sb1, 'square');
                sb1.GridLineStyle = 'none';
                sb1.XTick = [];
                sb1.YTick = [];
                cbp_lambda = colorbar('Parent', fig);
                sb2 = subplot(4, 3, 2 + (i - 1)*3, 'Parent', fig);
                if isempty(self.trainingData.cells)
                    self.trainingData.readData('c');
                end
                if any(self.modelParams.interpolationMode)
                    nx = numel(self.modelParams.gridSX) + 1;
                    ny = numel(self.modelParams.gridSY) + 1;
                    XX = reshape(self.trainingData.X_interp{1}(:, 1), nx, ny);
                    YY = reshape(self.trainingData.X_interp{1}(:, 2), nx, ny);
                    P = reshape(self.trainingData.P{i + dataOffset}, nx, ny);
                    trihandle = surf(XX, YY, P, 'Parent', sb2);
                else
                trihandle = trisurf(self.trainingData.cells{i + dataOffset},...
                    self.trainingData.X{i + dataOffset}(:, 1),...
                    self.trainingData.X{i + dataOffset}(:, 2),...
                    self.trainingData.P{i + dataOffset}, 'Parent', sb2);
                end
                trihandle.LineStyle = 'none';
                axis(sb2, 'tight');
                sb2.ZLim = [mean(self.trainingData.P{i + dataOffset}) - ...
                    3*std(self.trainingData.P{i + dataOffset}), ...
                    mean(self.trainingData.P{i + dataOffset}) + ...
                    3*std(self.trainingData.P{i + dataOffset})];
                caxis(sb2, sb2.ZLim);
                axis(sb2, 'square');
                sb2.View = [0, 90];
                sb2.GridLineStyle = 'none';
                sb2.XTick = [];
                sb2.YTick = [];
                sb2.Box = 'on';
                sb2.BoxStyle = 'full';
                
                cbp_true = colorbar('Parent', fig);
                
                sb3 = subplot(4, 3, 3 + (i - 1)*3, 'Parent', fig);
                D = zeros(2, 2, self.modelParams.coarseMesh.nEl);
                for j = 1:self.modelParams.coarseMesh.nEl
                    D(:, :, j) =  Lambda_eff_mode(j)*eye(2);
                end
                
                coarseFEMout = heat2d(self.modelParams.coarseMesh, D);
                
                Tc = coarseFEMout.Tff';
                Tc = Tc(:);
                if any(self.modelParams.interpolationMode)
                    reconstruction = reshape(self.modelParams.W_cf{1}*Tc,nx,ny);
                    trihandle2 = surf(XX, YY, reconstruction, 'Parent', sb3);
                else
                    reconstruction = self.modelParams.W_cf{i + dataOffset}*Tc;
                    trihandle2 =...
                        trisurf(self.trainingData.cells{i + dataOffset},...
                        self.trainingData.X{i + dataOffset}(:, 1),...
                        self.trainingData.X{i + dataOffset}(:, 2),...
                        reconstruction, 'Parent', sb3);
                end
                trihandle2.LineStyle = 'none';
                trihandle2.FaceColor = 'b';
                hold(sb3, 'on');
                if any(self.modelParams.interpolationMode)
                    trihandle3 = surf(XX, YY, P, 'Parent', sb3);
                else
                    trihandle3 =...
                        trisurf(self.trainingData.cells{i + dataOffset},...
                        self.trainingData.X{i + dataOffset}(:, 1),...
                        self.trainingData.X{i + dataOffset}(:, 2),...
                        self.trainingData.P{i + dataOffset}, 'Parent', sb3);
                end
                trihandle3.LineStyle = 'none';
                hold(sb3, 'off');
                axis(sb3, 'tight');
%                 sb3.ZLim = [mean(self.trainingData.P{i + dataOffset}) - ...
%                     3*std(self.trainingData.P{i + dataOffset}), ...
%                     mean(self.trainingData.P{i + dataOffset}) + ...
%                     3*std(self.trainingData.P{i + dataOffset})];
                caxis(sb3, sb3.ZLim);
                axis(sb3, 'square');
                sb3.Box = 'on';
                sb3.BoxStyle = 'full';
                sb3.XTick = [];
                sb3.YTick = [];
                cbp_reconst = colorbar('Parent', fig);
            end
            drawnow
        end
        
        function [predMeanArray, predVarArray, meanEffCond, meanSqDist,...
                sqDist, meanLogLikelihood] = predict(self, testStokesData, mode)
            %Function to predict finescale output from generative model
            %stokesData is a StokesData object of fine scale data
            %   mode:       'local' for separate theta_c's per macro-cell
            
            if nargin < 3
                mode = '';
            end
            
            
            %Some hard-coded prediction params
            nSamples_p_c = 1000;    %Samples
            
            
            %Load test data
            if isempty(testStokesData.X)
                testStokesData.readData('x');
            end
            if isempty(testStokesData.P)
                testStokesData.readData('p');
            end
            
            %if isempty(self.modelParams)
            if true
                %Read in trained params form ./data folder
                self.modelParams = ModelParams;
                self.modelParams.load;
                self.modelParams.rf2fem = self.modelParams.gridRF.map2fine(...
                    self.modelParams.coarseMesh.gridX,...
                    self.modelParams.coarseMesh.gridY);
            end
            
            testStokesData.evaluateFeatures(self.modelParams.gridRF);
            if exist('./data/featureFunctionMin', 'file')
                featFuncMin = dlmread('./data/featureFunctionMin');
                featFuncMax = dlmread('./data/featureFunctionMax');
                testStokesData.rescaleDesignMatrix(featFuncMin, featFuncMax);
            end
            if strcmp(mode, 'local')
                testStokesData.shapeToLocalDesignMat;
            end
            
            
            %% Sample from p_c
            disp('Sampling effective diffusivities...')
            nTest = numel(testStokesData.samples);
            
            %short hand notation/avoiding broadcast overhead
            nElc = self.modelParams.gridRF.nCells;
            
            Xsamples = zeros(nElc, nSamples_p_c, nTest);
            %Lambda as cell array for parallelization
            LambdaSamples{1} = zeros(nElc, nSamples_p_c);
            LambdaSamples = repmat(LambdaSamples, nTest, 1);
            
            meanEffCond = zeros(self.modelParams.coarseMesh.nEl, nTest);
            
            stdLogS = [];   %for parfor
            
            %to use point estimate for theta
%             self.modelParams.Sigma_theta_c =...
%                 1e-6*eye(numel(self.modelParams.gamma));
%             eig(self.modelParams.Sigma_theta_c)
%             pause
%             self.modelParams.Sigma_c = 1e-6*self.modelParams.Sigma_c;
%             self.modelParams.Sigma_c
%             self.modelParams.Sigma_c(self.modelParams.Sigma_c > 1e-1) = ...
%                 1e-1;
%             pause
            for i = 1:nTest
                if(strcmp(self.modelParams.prior_theta_c, 'VRVM') || ...
                        strcmp(self.modelParams.prior_theta_c, 'sharedVRVM'))
                    SigmaTildeInv = testStokesData.designMatrix{i}'*...
                        (self.modelParams.Sigma_c\...
                        testStokesData.designMatrix{i}) + ...
                        inv(self.modelParams.Sigma_theta_c);
                    
                    lastwarn('');
                    SigmaTilde = inv(SigmaTildeInv);
                    [~, id] = lastwarn; %to catch badly conditioned
                    if strcmp(id, 'MATLAB:nearlySingularMatrix')
                        mu_lambda_c = testStokesData.designMatrix{i}*...
                            self.modelParams.theta_c;
                        Sigma_lambda_c = self.modelParams.Sigma_c;
                    else
                        
                        Sigma_c_inv_Phi = self.modelParams.Sigma_c\...
                            testStokesData.designMatrix{i};
                        
                        precisionLambda_c = inv(self.modelParams.Sigma_c) - ...
                            Sigma_c_inv_Phi*(SigmaTildeInv\Sigma_c_inv_Phi');
                        
                        Sigma_lambda_c = inv(precisionLambda_c);
                        
                        mu_lambda_c = (precisionLambda_c\Sigma_c_inv_Phi)*...
                            (SigmaTilde/self.modelParams.Sigma_theta_c)*...
                            self.modelParams.theta_c;
                    end
                else
                    mu_lambda_c = testStokesData.designMatrix{i}*...
                    self.modelParams.theta_c;
                    Sigma_lambda_c = self.modelParams.Sigma_c;
                end
                %Samples of lambda_c
                if ~issymmetric(Sigma_lambda_c)
                    warning('Sigma_lambda_c not sym. to machine accuracy');
                    skew_sum = sum(sum(abs(...
                        .5*(Sigma_lambda_c - Sigma_lambda_c'))))
                    Sigma_lambda_c = .5*(Sigma_lambda_c + Sigma_lambda_c');
                end
                Sigma_lambda_c = 1e-12*eye(size(Sigma_lambda_c));
                Xsamples(:, :, i) = mvnrnd(mu_lambda_c',...
                    Sigma_lambda_c, nSamples_p_c)';
                %Diffusivities
                LambdaSamples{i} = conductivityBackTransform(...
                    Xsamples(:, :, i), self.modelParams.condTransOpts);
                meanEffCond(:, i) = self.modelParams.rf2fem*...
                    mean(LambdaSamples{i}, 2);
                LambdaSamples{i} = self.modelParams.rf2fem*LambdaSamples{i};
            end
            disp('Sampling from p_c done.')
            
            
            %% Run coarse model and sample from p_cf
            disp('Solving coarse model and sample from p_cf...')
            intp = any(self.modelParams.interpolationMode);
            if intp
                testStokesData.interpolate(self.modelParams.gridSX,...
                    self.modelParams.gridSY,...
                    self.modelParams.interpolationMode, ...
                    self.modelParams.smoothingParameter, ...
                    self.modelParams.boundarySmoothingPixels);
                testStokesData.shiftData(true);%p = 0 at orig. otherwise remove!
            else
                testStokesData.shiftData(false);
            end
            
            for n = 1:nTest
                predMeanArray{n} = zeros(size(testStokesData.P{n}));
            end
            predVarArray = predMeanArray;
            mean_squared_response = predMeanArray;
            
            cm = self.modelParams.coarseMesh;
            %Compute shape function interpolation matrices W
            if intp
                self.modelParams.fineScaleInterp(testStokesData.X_interp);
            else
                self.modelParams.fineScaleInterp(testStokesData.X);
            end
            W_cf = self.modelParams.W_cf;
            %S_n is a vector of variances at vertices
            testStokesData.vtxToCell(self.modelParams.gridSX,...
                self.modelParams.gridSY, self.modelParams.interpolationMode);
            P = testStokesData.P;
            if intp
                S = self.modelParams.sigma_cf.s0;
            else
                for n = 1:nTest
                    S{n} = self.modelParams.sigma_cf.s0(...
                        testStokesData.cellOfVertex{n});
                end
            end
            
            for n = 1:nTest
                for i = 1:nSamples_p_c
                    D = zeros(2, 2, cm.nEl);
                    for e = 1:cm.nEl
                        D(:, :, e) = LambdaSamples{n}(e, i)*eye(2);
                    end
                    FEMout = heat2d(cm, D);
                    Tctemp = FEMout.Tff';
                    
                    %sample from p_cf
                    if intp
                        mu_cf = W_cf{1}*Tctemp(:);
                    else
                        mu_cf = W_cf{n}*Tctemp(:);
                    end
                    
                    %only for diagonal S!!
                    %Sequentially compute mean and <Tf^2> to save memory
                    predMeanArray{n} = ((i - 1)/i)*predMeanArray{n}...
                        + (1/i)*mu_cf;  %U_f-integration can be done analyt.
                    mean_squared_response{n} = ((i - 1)/i)*...
                        mean_squared_response{n} + (1/i)*mu_cf.^2;
                end
                if intp
                    mean_squared_response{n} = mean_squared_response{n} + S;
                else
                    mean_squared_response{n} = mean_squared_response{n} + S{n};
                end
                
                %abs to avoid negative variance due to numerical error
                predVarArray{n} = abs(mean_squared_response{n...
                    } - predMeanArray{n}.^2);
                %meanTf_meanMCErr = mean(sqrt(predVarArray{n}/nSamples))
                
                %Remove response of first vertex as this is an essential node
                %ATTENTION: THIS IS ONLY VALID FOR B.C.'s WHERE VERTEX 1 IS THE 
                %ONLY ESSENTIAL VERTEX
                meanMahaErrTemp{n} =...
                    mean(sqrt(abs((1./(predVarArray{n}(2:end))).*...
                    (P{n}(2:end) - predMeanArray{n}(2:end)).^2)));
                sqDist{n} = (P{n}(2:end) - predMeanArray{n}(2:end)).^2;
                meanSqDistTemp{n} = mean(sqDist{n});    %mean over vertices
                
                logLikelihood{n} = -.5*numel(P{n}(2:end))*log(2*pi) -...
                    .5*sum(log(predVarArray{n}(2:end)), 'omitnan') - ...
                    .5*sum(sqDist{n}./predVarArray{n}(2:end), 'omitnan');
                %average over dof's
                meanLogLikelihood(n) = logLikelihood{n}/numel(P{n}(2:end));
                logPerplexity{n} = -(1/(numel(P{n}(2:end))))*logLikelihood{n};
            end
            
            meanMahalanobisError = mean(cell2mat(meanMahaErrTemp));
            meanSqDist = cell2mat(meanSqDistTemp);
            meanSqDistSq = mean(cell2mat(meanSqDistTemp).^2);
%             meanSquaredDistanceError =...
%                 sqrt((meanSqDistSq - meanSqDist^2)/nTest);
            meanLogPerplexity = mean(cell2mat(logPerplexity));
            meanPerplexity = exp(meanLogPerplexity);
            
            plotPrediction = true;
            if plotPrediction
                fig = figure('units','normalized','outerposition',[0 0 1 1]);
                pltstart = 0;
                if(isempty(testStokesData.cells) && ~intp)
                    testStokesData.readData('c');
                end
                for i = 1:6
                    %truth
                    splt(i) = subplot(2, 3, i);
                    if intp
                        nSX = numel(self.modelParams.gridSX) + 1;
                        nSY = numel(self.modelParams.gridSY) + 1;
                        XX = reshape(testStokesData.X_interp{1}(:, 1), nSX,nSY);
                        YY = reshape(testStokesData.X_interp{1}(:, 2), nSX,nSY);
                        P = reshape(testStokesData.P{i + pltstart}, nSX, nSY);
                        thdl = surf(XX, YY, P, 'Parent', splt(i));
                    else
                        thdl = trisurf(testStokesData.cells{i + pltstart},...
                            testStokesData.X{i + pltstart}(:, 1),...
                            testStokesData.X{i + pltstart}(:, 2),...
                            testStokesData.P{i + pltstart}, 'Parent', splt(i));
                    end
                    thdl.LineStyle = 'none';
                    axis(splt(i), 'tight');
                    axis(splt(i), 'square');
                    splt(i).View = [-80, 20];
                    splt(i).GridLineStyle = 'none';
%                     splt(i).XTick = [];
%                     splt(i).YTick = [];
                    splt(i).Box = 'on';
                    splt(i).BoxStyle = 'full';
                    splt(i).ZLim = [-2e4, 4e3];
                    cbp_true = colorbar('Parent', fig);
                    caxis = [min(testStokesData.P{i + pltstart}), ...
                        max(testStokesData.P{i + pltstart})];
                    
                    %predictive mean
                    hold on;
                    if intp
                        thdlpred = surf(XX, YY, reshape(...
                            predMeanArray{i + pltstart}, nSX, nSY),...
                            'Parent', splt(i));
                    else
                        thdlpred= trisurf(testStokesData.cells{i + pltstart},...
                            testStokesData.X{i + pltstart}(:, 1),...
                            testStokesData.X{i + pltstart}(:, 2),...
                            predMeanArray{i + pltstart}, 'Parent', splt(i));
                    end
                    thdlpred.LineStyle = 'none';
                    thdlpred.FaceColor = 'b';
                    
%                     %predictive mean + .5*std
%                     if intp
%                         thdlpstd = surf(XX, YY,...
%                             reshape(predMeanArray{i + pltstart} +...
%                             sqrt(predVarArray{i + pltstart}), nSX, nSY),...
%                             'Parent', splt(i));
%                     else
%                         thdlpstd= trisurf(testStokesData.cells{i + pltstart},...
%                             testStokesData.X{i + pltstart}(:, 1),...
%                             testStokesData.X{i + pltstart}(:, 2),...
%                             predMeanArray{i + pltstart} +...
%                             .5*sqrt(predVarArray{i + pltstart}),...
%                             'Parent', splt(i));
%                     end
%                     thdlpstd.LineStyle = 'none';
%                     thdlpstd.FaceColor = [.85 .85 .85];
%                     thdlpstd.FaceAlpha = .7;
%                     
%                     %predictive mean - .5*std
%                     if intp
%                         thdlmstd = surf(XX, YY,...
%                             reshape(predMeanArray{i + pltstart} -...
%                             sqrt(predVarArray{i + pltstart}), nSX, nSY),...
%                             'Parent', splt(i));
%                     else
%                         thdlmstd= trisurf(testStokesData.cells{i + pltstart},...
%                             testStokesData.X{i + pltstart}(:, 1),...
%                             testStokesData.X{i + pltstart}(:, 2),...
%                             predMeanArray{i + pltstart} -...
%                             .5*sqrt(predVarArray{i + pltstart}),...
%                             'Parent', splt(i));
%                     end
%                     thdlmstd.LineStyle = 'none';
%                     thdlmstd.FaceColor = [.85 .85 .85];
%                     thdlmstd.FaceAlpha = .7;

                end
            end
            
        end
        
        function [d_log_p_cf_sqMean] = findMeshRefinement(self)
            %Script to sample d_log_p_cf under Q(lambda_c) to find where to
            %refine mesh next
            
            if isempty(self.modelParams)
                %Read in trained params form ./data folder
                self.modelParams = ModelParams;
                self.modelParams = self.modelParams.load;
            end
            
            nSamples = 100;
            d_log_p_cf_sqMean = 0;
            k = 1;
            
            for n = (self.trainingData.samples + 1) %+1 for matlab indexing
                if(strcmp(self.modelParams.prior_theta_c, 'VRVM') || ...
                        strcmp(self.modelParams.prior_theta_c, 'sharedVRVM'))
                    SigmaTildeInv = self.trainingData.designMatrix{n}'*...
                        (self.modelParams.Sigma_c\...
                        self.trainingData.designMatrix{n}) + ...
                        inv(self.modelParams.Sigma_theta_c);
                    SigmaTilde = inv(SigmaTildeInv);
                    Sigma_c_inv_Phi = self.modelParams.Sigma_c\...
                        self.trainingData.designMatrix{n};
                    precisionLambda_c = inv(self.modelParams.Sigma_c) - ...
                        Sigma_c_inv_Phi*SigmaTilde*Sigma_c_inv_Phi';
                    Sigma_lambda_c = inv(precisionLambda_c);
                    mu_lambda_c = Sigma_lambda_c*Sigma_c_inv_Phi*(SigmaTilde/...
                       self.modelParams.Sigma_theta_c)*self.modelParams.theta_c;
                else
                    mu_lambda_c = self.trainingData.designMatrix{n}*...
                    self.modelParams.theta_c;
                    Sigma_lambda_c = self.modelParams.Sigma_c;
                end
                
                
                W_cf_n = self.modelParams.W_cf{n};
                %S_n is a vector of variances at vertices
                S_n = self.modelParams.sigma_cf.s0(...
                    self.trainingData.cellOfVertex{n});
                S_n = ones(size(S_n)); %comment to include S
                S_cf_n.sumLogS = sum(log(S_n));
                S_cf_n.Sinv_vec = 1./S_n;
                Sinv = sparse(1:length(S_n), 1:length(S_n), S_cf_n.Sinv_vec);
                S_cf_n.WTSinv = (Sinv*W_cf_n)';
                
                
                for j = 1:nSamples
                    lambda_c_sample = mvnrnd(mu_lambda_c, Sigma_lambda_c)';
                    [~, d_log_p_cf] = log_p_cf(self.trainingData.P{n},...
                        self.modelParams.coarseMesh, lambda_c_sample, W_cf_n,...
                        S_cf_n, self.modelParams.condTransOpts);

                    d_log_p_cf_sqMean = ((k - 1)/k)*d_log_p_cf_sqMean +...
                        (1/k)*d_log_p_cf.^2;
                    k = k + 1;
                end
            end
            
            fig = figure;
            sb1 = subplot(1, 1, 1);
            imagesc(reshape(d_log_p_cf_sqMean,...
                self.modelParams.coarseMesh.nElX,...
                self.modelParams.coarseMesh.nElY)', 'Parent', sb1)
            sb1.YDir = 'normal';
            axis(sb1, 'tight');
            axis(sb1, 'square');
            sb1.GridLineStyle = 'none';
            sb1.XTick = [];
            sb1.YTick = [];
            cbp_lambda = colorbar('Parent', fig);
        end
    end
end

