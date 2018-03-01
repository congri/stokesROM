classdef ModelParams < handle
    %Initialize, update, ... the ROM model params
    
    properties
        %% Model parameters
        %p_c
        theta_c
        Sigma_c
        %posterior variance of theta_c, given a prior model
        Sigma_theta_c
        gridRF
        
        %p_cf
        W_cf
        sigma_cf
        gridSX
        gridSY
        interpolationMode
        smoothingParameter
        
        %Surrogate FEM mesh
        coarseMesh
        
        %Mapping from random field to FEM discretization
        rf2fem
        
        %Transformation options of diffusivity parameter
        condTransOpts
        
        %% Model hyperparameters
        prior_theta_c = 'sharedVRVM'
        gamma   %Gaussian precision of prior on theta_c
        VRVM_a = 1e-10;
        VRVM_b = 1e-10;
        VRVM_c = 1e-10;
        VRVM_d = 1e-10;
        VRVM_e = 1e-10;
        VRVM_f = 1e-10;
        VRVM_iter = 200; %iterations with fixed q(lambda_c)
        
        %% Parameters of variational distributions
        variational_mu
        variational_sigma
        
        %% Parameters to rescale features
        featureFunctionMean
        featureFunctionSqMean
        featureFunctionMin
        featureFunctionMax
    end
    
    methods
        function self = ModelParams()
            %Constructor
        end
        
        function initialize(self, nElements, nData, mode)
            %Initialize model parameters
            %   nFeatures:      number of feature functions
            %   nElements:      number of macro elements
            %   nSCells:        number of cells in S-grid
            
            if strcmp(mode, 'load')
                self.load;
                
                %Initialize parameters of variational approximate distributions
                load('./data/vardistparams.mat');
                self.variational_mu{1} = varmu;
                self.variational_mu = varsigma;
                
                self.variational_sigma{1} = 1e0*ones(1, nElements);
                self.variational_sigma =...
                    repmat(self.variational_sigma, nData, 1);
            else
                %Initialize sigma_c to I
                self.Sigma_c = 1e-4*eye(nElements);
                
                nSX = numel(self.gridSX); nSY = numel(self.gridSY);
                if any(self.interpolationMode)
                    nSX = nSX + 1; nSY = nSY + 1;
                end
                self.sigma_cf.s0 = ones(nSX*nSY, 1);  %variance field of p_cf
                
                %Initialize hyperparameters
                if strcmp(self.prior_theta_c, 'RVM')
                    self.gamma = 1e-4*ones(size(self.theta_c));
                elseif strcmp(self.prior_theta_c, 'VRVM')
                    self.gamma = 1e-2*ones(size(self.theta_c));
                elseif strcmp(self.prior_theta_c, 'sharedVRVM')
                    self.gamma = 1e-2*ones(size(self.theta_c));
                elseif strcmp(self.prior_theta_c, 'none')
                    self.gamma = NaN;
                else
                    error('What prior model for theta_c?')
                end
                
                %Initialize parameters of variational approximate distributions
                self.variational_mu{1} = 0*ones(1, nElements);
                self.variational_mu = repmat(self.variational_mu, nData, 1);
                
                self.variational_sigma{1} = 1e0*ones(1, nElements);
                self.variational_sigma =...
                    repmat(self.variational_sigma, nData, 1);
            end
        end
        
        function load(self)
            %Initialize params theta_c, theta_cf
                        
            %Coarse mesh object
            if exist('./data/coarseMesh.mat', 'file')
                load('./data/coarseMesh.mat', 'coarseMesh');
                self.coarseMesh = coarseMesh;
            else
                error('No coarseMesh found. Gen. Mesh obj. and save to ./data')
            end
            
            %Load trained params from disk
            disp('Loading trained parameters from disk...')
            self.gamma = dlmread('./data/thetaPriorHyperparam');
            self.gamma = self.gamma(end, :);
            load('./data/prior_theta_c');
            self.prior_theta_c = priortype;
            self.theta_c = dlmread('./data/theta_c');
            self.theta_c = self.theta_c(end, :)';
            self.Sigma_c = dlmread('./data/sigma_c');
            self.Sigma_c = diag(self.Sigma_c(end, :));
            load('./data/condTransOpts.mat');
            self.condTransOpts = condTransOpts;

            addpath('./mesh');
            load('./data/gridRF.mat');
            self.gridRF = gridRF;
            self.rf2fem = gridRF.map2fine(coarseMesh.gridX, coarseMesh.gridY);
            
            self.sigma_cf.s0 = dlmread('./data/sigma_cf')';
            load('./data/gridS.mat');
            self.gridSX = gridSX;
            self.gridSY = gridSY;
            
            load('./data/interpolationMode.mat');
            self.interpolationMode = interpolationMode;
            
            load('./data/smoothingParameter.mat');
            self.smoothingParameter = smoothingParameter;
            
            try
                temp = dlmread('./data/Sigma_theta_c');
                %temp = temp(end, :);
                self.Sigma_theta_c = reshape(temp, sqrt(numel(temp)),...
                    sqrt(numel(temp)));
            catch
                warning('Sigma_theta_c not found.');
            end
            disp('done')
            
            disp('Loading data normalization data...')
            try
                self.featureFunctionMean =...
                    dlmread('./data/featureFunctionMean');
                self.featureFunctionSqMean =...
                    dlmread('./data/featureFunctionSqMean');
            catch
                warning(strcat('featureFunctionMean, featureFunctionSqMean',...
                    'not found, setting it to 0.'))
                self.featureFunctionMean = 0;
                self.featureFunctionSqMean = 0;
            end
            
            try
                self.featureFunctionMin = dlmread('./data/featureFunctionMin');
                self.featureFunctionMax = dlmread('./data/featureFunctionMax');
            catch
                warning(strcat('featureFunctionMin, featureFunctionMax', ...
                    'not found, setting it to 0.'))
                self.featureFunctionMin = 0;
                self.featureFunctionMax = 0;
            end
            disp('done')
        end
        
        function printCurrentParams(self, mode)
            %Print current model params on screen
            
            if strcmp(mode, 'local')
                disp('theta_c: row = feature, column = macro-cell:')
                curr_theta_c = reshape(self.theta_c,...
                    numel(self.theta_c)/self.gridRF.nCells,...
                    self.gridRF.nCells)
                curr_Sigma_c = full(diag(self.Sigma_c))
                if strcmp(self.prior_theta_c, 'sharedVRVM')
                    curr_gamma = self.gamma(1:...
                        (numel(self.theta_c)/self.gridRF.nCells))
                else
                    curr_gamma = self.gamma
                end
            else
                curr_theta_c = self.theta_c
                curr_Sigma_c = full(diag(self.Sigma_c))
                curr_gamma = self.gamma
            end
            
        end
        
        function fineScaleInterp(self, X)
            %Precompute shape function interp. on every fine scale vertex
            
            nData = numel(X);
            for n = 1:nData
                self.W_cf{n} = shapeInterp(self.coarseMesh, X{n});
            end
            
            self.saveParams('W');
        end
        
        function plot_params(self, figHandle, thetaArray, SigmaArray)
            %Plots the current theta_c
            
            %short notation
            nSX = numel(self.gridSX); nSY = numel(self.gridSY);
            if any(self.interpolationMode)
                nSX = nSX + 1; nSY = nSY + 1;
            end
            
            if nargin < 2
                figHandle = figure;
                if nargin < 3
                    thetaArray = dlmread('./data/theta_c');
                    if nargin < 4
                        SigmaArray = dlmread('./data/sigma_c');
                    end
                end
            end
            
            sb1 = subplot(3, 2, 1, 'Parent', figHandle);
            plot(thetaArray, 'linewidth', 1, 'Parent', sb1)
            axis(sb1, 'tight');
            sb1.YLim = [(min(thetaArray(end, :)) - 1),...
                (max(thetaArray(end, :)) + 1)];
            
            sb2 = subplot(3, 2, 2, 'Parent', figHandle);
            bar(self.theta_c, 'linewidth', 1, 'Parent', sb2)
            axis(sb2, 'tight');
            
            sb3 = subplot(3,2,3, 'Parent', figHandle);
            semilogy(sqrt(SigmaArray), 'linewidth', 1, 'Parent', sb3)
            axis(sb3, 'tight');
            sb3.XLabel.String = 'iter';
            sb3.YLabel.String = '$\sigma_k$';
            
            sb4 = subplot(3, 2, 4, 'Parent', figHandle);
            
            sigma_c_plot = self.rf2fem*diag(self.Sigma_c);
            im = imagesc(reshape(sigma_c_plot, self.coarseMesh.nElX,...
                self.coarseMesh.nElY)', 'Parent', sb4);
            sb4.YDir = 'normal';
            sb4.Title.String = '$\sigma_k$';
            colorbar('Parent', figHandle);
            sb4.GridLineStyle = 'none';
            axis(sb4, 'square');
            
            sb5 = subplot(3, 2, 5, 'Parent', figHandle);
            if strcmp(self.prior_theta_c, 'sharedVRVM')
                bar(self.gamma(1:(numel(self.theta_c)/self.gridRF.nCells)),...
                    'linewidth', 1, 'Parent', sb5)
            else
                bar(self.gamma, 'linewidth', 1, 'Parent', sb5)
            end
            axis(sb5, 'tight');
            sb5.YLabel.String = '$\gamma$';
            sb5.YScale = 'log';
            
            sb6 = subplot(3, 2, 6, 'Parent', figHandle);
            imagesc(reshape(sqrt(self.sigma_cf.s0), nSX, nSY)', 'Parent', sb6)
            sb6.Title.String = 'S';
            colorbar('Parent', figHandle);
            sb6.GridLineStyle = 'none';
            axis(sb6, 'square');
            sb6.YDir = 'normal';
        end
                
        function saveParams(self, params)
            if ~exist('./data/', 'dir')
                mkdir('./data/');
            end
            
            %coarseMesh
            if contains(params, 'coarseMesh')
                %save coarseMesh to file
                coarseMesh = self.coarseMesh;
                save('./data/coarseMesh.mat', 'coarseMesh');
            end
            
            %Coarse random field discretization
            if contains(params, 'gridRF')
                gridRF = self.gridRF;
                save('./data/gridRF.mat', 'gridRF');
            end
            
            %Optimal params
            %W matrices
            if any(params == 'W')
                for i = 1:numel(self.W_cf)
                    filename = strcat('./data/Wmat', num2str(i));
                    [rowW, colW, valW] = find(self.W_cf{i});
                    WArray = [rowW, colW, valW]';
                    save(filename, 'WArray', '-ascii')
                end
            end
            
            %prior type
            if contains(params, 'priorType')
                priortype = self.prior_theta_c;
                save('./data/prior_theta_c.mat', 'priortype');
            end
            
            %transformation options of effective diffusion field
            if contains(params, 'condTransOpts')
                condTransOpts = self.condTransOpts;
                save('./data/condTransOpts.mat', 'condTransOpts');
            end
            
            %gamma
            if any(params == 'g')
                filename = './data/thetaPriorHyperparam';
                thetaPriorHyperparam = self.gamma';
                save(filename, 'thetaPriorHyperparam', '-ascii', '-append');
            end
            
            %theta_c
            if contains(params, 'tc')
                filename = './data/theta_c';
                tc = self.theta_c';
                save(filename, 'tc', '-ascii', '-append');
            end
            
            %Sigma_theta_c (variance of posterior on theta_c)
            if contains(params, 'stc')
                filename = './data/Sigma_theta_c';
                stc = self.Sigma_theta_c(:)';
                onlyFinal = true;
                if onlyFinal
                    save(filename, 'stc', '-ascii');
                else
                    save(filename, 'stc', '-ascii', '-append');
                end
            end
            
            %sigma
            if contains(params, 'sc')
                filename = './data/sigma_c';
                sc = full(diag(self.Sigma_c))';
                save(filename, 'sc', '-ascii', '-append');
            end
            
            %S
            if contains(params, 'scf')
                filename = './data/sigma_cf';
                scf = self.sigma_cf.s0';
                onlyFinal = true;
                if onlyFinal
                    save(filename, 'scf', '-ascii');
                else
                    save(filename, 'scf', '-ascii', '-append');
                end
            end
            
            %grid of S
            if contains(params, 'gridS')
                filename = './data/gridS.mat';
                gridSX = self.gridSX;
                gridSY = self.gridSY;
                save(filename, 'gridSX', 'gridSY');
            end
            
            %Interpolation mode
            if contains(params, 'interp')
                interpolationMode = self.interpolationMode;
                save('./data/interpolationMode.mat', 'interpolationMode');
            end
            
            %Smoothing parameter for interpolated data
            if contains(params, 'smooth')
                smoothingParameter = self.smoothingParameter;
                save('./data/smoothingParameter.mat', 'smoothingParameter');
            end
            
            %Parameters of variational distributions on log lambda_c
            if contains(params, 'vardist')
                varmu = self.variational_mu;
                varsigma = self.variational_sigma;
                save('./data/vardistparams.mat', 'varmu', 'varsigma');
            end
            
        end
    end
end

