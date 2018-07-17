classdef StokesROM < handle
    %Class for reduced order model of Stokes equation
    
    properties
        
        %StokesData object
        trainingData
        
        %Model parameters
        modelParams
        
    end
    
    methods
        function self = StokesROM()
            %Constructor
        end
        
        function M_step(self, XMean, XSqMean, sqDist_p_cf)
            
            if(strcmp(self.modelParams.prior_theta_c, 'VRVM') || ...
                    strcmp(self.modelParams.prior_theta_c, 'sharedVRVM'))
                dim_theta = numel(self.modelParams.theta_c);
                nElc = size(self.trainingData.designMatrix{1}, 1);
                
                %Parameters that do not change when q(lambda_c) is fixed
                a = self.modelParams.VRVM_a + .5;
                e = self.modelParams.VRVM_e + .5*self.trainingData.nSamples;
                c = self.modelParams.VRVM_c + .5*self.trainingData.nSamples;
                Ncells_gridS = numel(self.modelParams.fineGridX)*...
                    numel(self.modelParams.fineGridY);
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
                gam = self.modelParams.gamma;
                tau_theta = diag(gam);    %precision of q(theta_c)
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
                    gam = a./b;
                    d = self.modelParams.VRVM_d + .5*sum(XSqMean, 2);
                    for n = 1:self.trainingData.nSamples
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
                    if strcmp(self.modelParams.mode, 'local')
                        nFeatures = dim_theta/nElc;
                        tau_theta = sparse(1:dim_theta, 1:dim_theta, gam,...
                            dim_theta, dim_theta, nElc*nFeatures^2);
                    else
                        tau_theta = diag(gam);
                    end
                    sumPhiTau_cXMean = 0;
                    for n = 1:self.trainingData.nSamples
                        %to ensure pos. def.
                        %A = diag(sqrt_tau_c)*self.trainingData.designMatrix{n};
                        A = sqrt_tau_c.*self.trainingData.designMatrix{n};
                        tau_theta = tau_theta + A'*A;
%                         tau_theta = tau_theta +...
%                             self.trainingData.designMatrix{n}'*diag(tau_c)*...
%                             self.trainingData.designMatrix{n};
                        sumPhiTau_cXMean = sumPhiTau_cXMean + ...
                            self.trainingData.designMatrix{n}'*...
                            diag(tau_c)*XMean(:, n);
                    end
                    
                    if(strcmp(self.modelParams.mode, 'local') && nElc > 4)
                        %solve block-diagonal tau_theta
                        %break-even at nElc == 4?
                        nFeatures = dim_theta/nElc;
                        Sigma_theta = spalloc(dim_theta, dim_theta, ...
                            nElc*nFeatures^2);
                        %can even be done in parallel
                        for k = 1:nElc
                            Sigma_theta(((k-1)*nFeatures + 1):(k*nFeatures),...
                                ((k - 1)*nFeatures + 1):(k*nFeatures)) = inv(...
                                tau_theta(((k-1)*nFeatures+ 1):(k*nFeatures),...
                                ((k - 1)*nFeatures + 1):(k*nFeatures)));
                        end
                    else
                        Sigma_theta = inv(tau_theta);
                    end
                    mu_theta = Sigma_theta*sumPhiTau_cXMean;
                end
                
                %assign <S>, <Sigma_c>, <theta_c>
                self.modelParams.sigma_cf.s0 = 1./tau_cf;
                
                self.modelParams.Sigma_c = diag(1./tau_c);
                self.modelParams.theta_c = mu_theta;
                self.modelParams.Sigma_theta_c = Sigma_theta;
                
                self.modelParams.gamma = gam;
                self.modelParams.a = a;    self.modelParams.b = b;
                self.modelParams.c = c;    self.modelParams.d = d;
                self.modelParams.e = e;    self.modelParams.f = f;
                mean_s0 = mean(self.modelParams.sigma_cf.s0)
            else
                %Update model parameters
                self.update_p_c(XMean, XSqMean);
                self.update_p_cf(sqDist_p_cf);
            end
        end
        
        function update_p_c(self, XMean, XSqMean)
            %% Find optimal theta_c and Sigma_c self-consistently:
            %update theta_c, then Sigma_c, then theta_c and so on
            
            %short-hand notation
            dim_theta_c = numel(self.modelParams.theta_c);
            N_train = self.trainingData.nSamples;
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
            
            Ncells_gridS = numel(self.modelParams.fineGridX)*...
                numel(self.modelParams.fineGridY);
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
                
        function plotCurrentState(self, fig, dataOffset, transType, transLimits)
            %Plots the current modal effective property and the modal
            %reconstruction for 2 -training- samples
            for i = 1:4
                Lambda_eff_mode = conductivityBackTransform(...
                    self.trainingData.designMatrix{i + dataOffset}*...
                    self.modelParams.theta_c, transType, transLimits);
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
                if isempty(sb2.Children)
                    th = trisurf(self.trainingData.cells{i + dataOffset},...
                        self.trainingData.X{i + dataOffset}(:, 1),...
                        self.trainingData.X{i + dataOffset}(:, 2),...
                        zeros(size(self.trainingData.X{i + dataOffset}, 1),1),...
                        'Parent', sb2);
                    th.LineStyle = 'none';
                    axis(sb2, 'tight');
                    axis(sb2, 'square');
                    sb2.View = [0, 90];
                    sb2.GridLineStyle = 'none';
                    sb2.XTick = [];
                    sb2.YTick = [];
                    sb2.Box = 'on';
                    sb2.BoxStyle = 'full';
                    th.FaceColor = 'k';
                end
                
%                 cbp_true = colorbar('Parent', fig);
                
                sb3 = subplot(4, 3, 3 + (i - 1)*3, 'Parent', fig);
                
                isotropicDiffusivity = true;
                if isotropicDiffusivity
                    coarseFEMout =...
                        heat2d(self.modelParams.coarseMesh, Lambda_eff_mode);
                else
                    D = zeros(2, 2, self.modelParams.coarseMesh.nEl);
                    for j = 1:self.modelParams.coarseMesh.nEl
                        D(:, :, j) =  Lambda_eff_mode(j)*eye(2);
                    end
                    coarseFEMout = heat2d(self.modelParams.coarseMesh, D);
                end
                
                Tc = coarseFEMout.Tff';
                Tc = Tc(:);
                nx = numel(self.modelParams.fineGridX) + 1;
                ny = numel(self.modelParams.fineGridY) + 1;
                XX = reshape(self.trainingData.X_interp{1}(:, 1), nx, ny);
                YY = reshape(self.trainingData.X_interp{1}(:, 2), nx, ny);
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
                    P = reshape(self.trainingData.P{i + dataOffset}, nx, ny);
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
                sqDist, meanLogLikelihood, R, R_i] =...
                predict(self, testData, mode)
            %Function to predict finescale output from generative model
            %stokesData is a StokesData object of fine scale data
            %   mode:       'local' for separate theta_c's per macro-cell
            
            if nargin < 3
                mode = '';
            end
            
            %Some hard-coded prediction params
            nSamples_p_c = 1000;    %Samples
            
            %Load test data
            if isempty(testData.X)
                testData.readData('x');
            end
            if isempty(testData.P)
                testData.readData('p');
            end
            
            %Read in trained params form ./data folder
            disp('Loading trained model params...')
            load('./data/modelParams.mat');
            self.modelParams = modelParams;
            disp('...trained model params loaded.')
            
            testData.evaluateFeatures(self.modelParams.gridRF);
            if exist('./data/featureFunctionMin', 'file')
                featFuncMin = dlmread('./data/featureFunctionMin');
                featFuncMax = dlmread('./data/featureFunctionMax');
                testData.rescaleDesignMatrix(featFuncMin, featFuncMax);
            end
            if strcmp(mode, 'local')
                testData.shapeToLocalDesignMat;
            end
            
            
            %% Sample from p_c
            disp('Sampling effective diffusivities...')
            nTest = numel(testData.samples);
            
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
            tau_theta_c = inv(self.modelParams.Sigma_theta_c);
            Sigma_c_vec_inv = 1./diag(self.modelParams.Sigma_c);
            for i = 1:nTest
                if(strcmp(self.modelParams.prior_theta_c, 'VRVM') || ...
                        strcmp(self.modelParams.prior_theta_c, 'sharedVRVM'))
                    SigmaTildeInv = testData.designMatrix{i}'*...
                        (Sigma_c_vec_inv.*testData.designMatrix{i}) +...
                        tau_theta_c;
                    
                    lastwarn('');
                    SigmaTilde = inv(SigmaTildeInv);
                    [~, id] = lastwarn; %to catch badly conditioned
                    if strcmp(id, 'MATLAB:nearlySingularMatrix')
                        mu_lambda_c = testData.designMatrix{i}*...
                            self.modelParams.theta_c;
                        Sigma_lambda_c = self.modelParams.Sigma_c;
                    else
                        
                        Sigma_c_inv_Phi = self.modelParams.Sigma_c\...
                            testData.designMatrix{i};
                        
                        precisionLambda_c = inv(self.modelParams.Sigma_c) - ...
                            Sigma_c_inv_Phi*(SigmaTildeInv\Sigma_c_inv_Phi');
                        
                        Sigma_lambda_c = inv(precisionLambda_c);
                        
                        mu_lambda_c = (precisionLambda_c\Sigma_c_inv_Phi)*...
                            (SigmaTilde/self.modelParams.Sigma_theta_c)*...
                            self.modelParams.theta_c;
                    end
                else
                    mu_lambda_c = testData.designMatrix{i}*...
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
                    Xsamples(:, :, i), self.modelParams.diffTransform,...
                    self.modelParams.diffLimits);
                meanEffCond(:, i) = self.modelParams.rf2fem*...
                    mean(LambdaSamples{i}, 2);
                LambdaSamples{i} = self.modelParams.rf2fem*LambdaSamples{i};
            end
            disp('Sampling from p_c done.')
            
            
            %% Run coarse model and sample from p_cf
            disp('Solving coarse model and sample from p_cf...')
            intp = any(self.modelParams.interpolationMode);
            if intp
                testData.interpolate(self.modelParams);
                testData.shiftData(true);%p = 0 at orig. otherwise remove!
            else
                testData.shiftData(false);
            end
            
            for n = 1:nTest
                predMeanArray{n} = zeros(size(testData.P{n}));
            end
            predVarArray = predMeanArray;
            mean_squared_response = predMeanArray;
            
            cm = self.modelParams.coarseMesh;
            %Compute shape function interpolation matrices W
            if intp
                self.modelParams.fineScaleInterp(testData.X_interp);
            else
                self.modelParams.fineScaleInterp(testData.X);
            end
            W_cf = self.modelParams.W_cf;
            %S_n is a vector of variances at vertices
            testData.vtx2Cell(self.modelParams);
            P = testData.P;
            if intp
                S = self.modelParams.sigma_cf.s0;
            else
                for n = 1:nTest
                    S{n} = self.modelParams.sigma_cf.s0(...
                        testData.cellOfVertex{n});
                end
            end
            
            parfor n = 1:nTest
                for i = 1:nSamples_p_c
                    
                    isotropicDiffusivity = true;
                    if isotropicDiffusivity
                        FEMout = heat2d(cm, LambdaSamples{n}(:, i));
                    else
                        D = zeros(2, 2, cm.nEl);
                        for e = 1:cm.nEl
                            D(:, :, e) = LambdaSamples{n}(e, i)*eye(2);
                        end
                        FEMout = heat2d(cm, D);
                    end
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
            meanSqDist = mean(cell2mat(sqDist), 2);
            meanSqDistSq = mean(cell2mat(meanSqDistTemp).^2);
%             meanSquaredDistanceError =...
%                 sqrt((meanSqDistSq - meanSqDist^2)/nTest);
            meanLogPerplexity = mean(cell2mat(logPerplexity));
            meanPerplexity = exp(meanLogPerplexity);
            
            %Coefficient of determination, see wikipedia
            SS_res = mean(meanSqDist);
            P = cell2mat(P);
            p_bar = mean(P, 2);
            p_bar_bar = mean(p_bar)
            p_var = mean((P - p_bar).^2, 2);
            p_var_bar = mean(p_var)
            SS_tot = mean(var(P, 1, 2));
            R = 1 - SS_res/SS_tot;
            R_i = 1 - meanSqDist./p_var(2:end);

            
            %% plotting the predictions
            plotPrediction = true;
            if plotPrediction
                fig = figure('units','normalized','outerposition',[0 0 1 1]);
                pltstart = 0;
                if(isempty(testData.cells) && ~intp)
                    testData.readData('c');
                end
                for i = 1:6
                    %truth
                    splt(i) = subplot(2, 3, i);
                    if intp
                        nSX = numel(self.modelParams.fineGridX) + 1;
                        nSY = numel(self.modelParams.fineGridY) + 1;
                        XX = reshape(testData.X_interp{1}(:, 1), nSX,nSY);
                        YY = reshape(testData.X_interp{1}(:, 2), nSX,nSY);
                        P = reshape(testData.P{i + pltstart}, nSX, nSY);
                        thdl = surf(XX, YY, P, 'Parent', splt(i));
                    else
                        thdl = trisurf(testData.cells{i + pltstart},...
                            testData.X{i + pltstart}(:, 1),...
                            testData.X{i + pltstart}(:, 2),...
                            testData.P{i + pltstart}, 'Parent', splt(i));
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
%                     splt(i).ZLim = [-2e4, 4e3];
                    cbp_true = colorbar('Parent', fig);
                    caxis = [min(testData.P{i + pltstart}), ...
                        max(testData.P{i + pltstart})];
                    
                    %predictive mean
                    hold on;
                    if intp
                        thdlpred = surf(XX, YY, reshape(...
                            predMeanArray{i + pltstart}, nSX, nSY),...
                            'Parent', splt(i));
                    else
                        thdlpred= trisurf(testData.cells{i + pltstart},...
                            testData.X{i + pltstart}(:, 1),...
                            testData.X{i + pltstart}(:, 2),...
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
        
        function [d_log_p_cf_sqMean] = findMeshRefinement(self, include_S)
            %Script to sample d_log_p_cf under Q(lambda_c) to find where to
            %refine mesh next
            %    include_S:    use trained S or S = 1 in log p_cf
            
            if isempty(self.modelParams)
                %Read in trained params form ./data folder
                %self.modelParams = ModelParams;
                %self.modelParams = self.modelParams.load;
                load('./data/modelParams.mat');
                self.modelParams = modelParams;     clear modelParams;
            end
            
            MCsamples = 2000;
            d_log_p_cf_sqMean{1} = 0;
            d_log_p_cf_sqMean =...
                repmat(d_log_p_cf_sqMean, 1, self.trainingData.nSamples);
            
            if isempty(self.modelParams.interpolationMode)
                error('Only implemented for interpolation on regular grid.')
            else
                W_cf_n = self.modelParams.W_cf{1};
                %S_n is a vector of variances at vertices
                S_n = self.modelParams.sigma_cf.s0;
                if ~include_S
                    S_n = ones(size(S_n));
                end
                S_cf_n.sumLogS = sum(log(S_n));
                S_cf_n.Sinv_vec = 1./S_n;
                Sinv =sparse(1:length(S_n), 1:length(S_n), S_cf_n.Sinv_vec);
                S_cf_n.WTSinv = (Sinv*W_cf_n)';
            end
            
            %copying for parfor memory efficiency
            nSamples = self.trainingData.nSamples;
            prior_theta_c = self.modelParams.prior_theta_c;
            designMatrix = self.trainingData.designMatrix;
            Sigma_c = self.modelParams.Sigma_c;
            Sigma_theta_c = self.modelParams.Sigma_theta_c;
            theta_c = self.modelParams.theta_c;
            P = self.trainingData.P;
            cm = self.modelParams.coarseMesh;
            diffTransform = self.modelParams.diffTransform;
            diffLimits = self.modelParams.diffLimits;
            rf2fem = self.modelParams.rf2fem;
            
            tic;
            parfor n = 1:nSamples
                if(strcmp(prior_theta_c, 'VRVM') || ...
                        strcmp(prior_theta_c, 'sharedVRVM'))
                    SigmaTildeInv = designMatrix{n}'*...
                        (Sigma_c\designMatrix{n}) + ...
                        inv(Sigma_theta_c);
                    SigmaTilde = inv(SigmaTildeInv);
                    Sigma_c_inv_Phi = Sigma_c\designMatrix{n};
                    precisionLambda_c = inv(Sigma_c) - ...
                        Sigma_c_inv_Phi*SigmaTilde*Sigma_c_inv_Phi';
                    Sigma_lambda_c = inv(precisionLambda_c);
                    mu_lambda_c = Sigma_lambda_c*Sigma_c_inv_Phi*(SigmaTilde/...
                       Sigma_theta_c)*theta_c;
                else
                    mu_lambda_c = designMatrix{n}*theta_c;
                    Sigma_lambda_c = Sigma_c;
                end

                
                for j = 1:MCsamples
%                     lambda_c_sample = mvnrnd(mu_lambda_c, Sigma_lambda_c)';
                    lambda_c_sample = normrnd(mu_lambda_c,...
                        sqrt(diag(Sigma_lambda_c)));
%                     lambda_c_sample = normrnd(...
%                         self.modelParams.variational_mu{n}, ...
%                         self.modelParams.variational_sigma{n})';
                    [~, d_log_p_cf] = log_p_cf(P{n},cm, lambda_c_sample,...
                        W_cf_n, S_cf_n, diffTransform, diffLimits, rf2fem,...
                        true);

                    d_log_p_cf_sqMean{n} = ((j - 1)/j)*d_log_p_cf_sqMean{n} +...
                        (1/j)*d_log_p_cf.^2;
                end
            end
            t_activeCells = toc
            d_log_p_cf_sqMean = cell2mat(d_log_p_cf_sqMean);
            d_log_p_cf_sqMean = mean(d_log_p_cf_sqMean, 2);
        end
    end
end

