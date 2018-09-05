classdef ModelParams < matlab.mixin.Copyable
    %Initialize, update, ... the ROM model params
    
    properties
        %% Model parameters
        %p_c
        theta_c
        Sigma_c
        %posterior variance of theta_c, given a prior model
        Sigma_theta_c
        %FEM grid of coarse Darcy emulator
        coarseGridX = (1/16)*ones(1, 16)
        coarseGridY = (1/16)*ones(1, 16)
        %grid of random field
        gridRF
        splitted_cells
        
        %Matrix summing up all values of a fine-scale vector belonging to
        %a certain macro-cell
        sum_in_macrocell
        
        %Recorded elbo
        elbo
        cell_score
        cell_score_full   %includes also terms of p_cf to elbo cell score
        
        %p_cf
        W_cf
        sigma_cf
        fineGridX = (1/128)*ones(1, 128)
        fineGridY = (1/128)*ones(1, 128)
        interpolationMode = 'nearest'   %'nearest', 'linear' or 'natural'
        smoothingParameter = []
        boundarySmoothingPixels = -1   %only smooths boundary if positive
        
        %Surrogate FEM mesh
        coarseMesh
        
        %Mapping from random field to FEM discretization
        rf2fem
        %cell index dictionary
        cell_dictionary
        
        %Transformation options of diffusivity parameter
        diffTransform = 'log'
        diffLimits = [1e-10, 1e8];
        
        %% Model hyperparameters
        mode = 'local'  %separate theta_c's per macro-cell
        prior_theta_c = 'sharedVRVM'
        gamma   %Gaussian precision of prior on theta_c
        VRVM_a = eps
        VRVM_b = eps
        VRVM_c = 1e-6
        VRVM_d = eps
        VRVM_e = eps
        VRVM_f = eps
        VRVM_iter = 10 %iterations with fixed q(lambda_c)
        
        %current parameters of variational distributions
        a
        b
        c
        d
        e
        f
        
        %% Parameters of variational distributions
        varDistParamsVec
        variational_mu
        variational_sigma
        
        %% Parameters to rescale features
        normalization = 'rescale'
        featureFunctionMean
        featureFunctionSqMean
        featureFunctionMin
        featureFunctionMax
        
        %% Training parameters
        max_EM_iter = 50
        
        %% Settings
        computeElbo = true
        
        %% plots, initialized as animated lines
        p_theta
        p_sigma
        p_gamma
        p_elbo
    end
    
    methods
        function self = ModelParams(u_bc, p_bc)
            %Constructor; Set up all params that are unchanged during training
            %   u_bc:       boundary velocity field
            %   p_bc:       boundary pressure field
            
            %only for a single cell here!!!
            %grid of random field
            self.gridRF = RectangularMesh((1/2)*ones(1, 2));
            self.cell_dictionary = 1:self.gridRF.nCells;
            
            %% Initialize coarse mesh object
            %Coarse mesh object
            self.coarseMesh = MeshFEM(self.coarseGridX, self.coarseGridY);
            self.coarseMesh.compute_grad = true;
            
            %% Set up coarse model bc's
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
            
            p_bc_handle = str2func(strcat('@(x)', p_bc));
            
            nX = length(self.coarseGridX);
            nY = length(self.coarseGridY);
            self.coarseMesh = self.coarseMesh.setBoundaries(2:(2*nX + 2*nY),...
                p_bc_handle, u_bc_handle);
        end
        
        function initialize(self, nData)
            %Initialize model parameters
            %   nFeatures:      number of feature functions
            %   nElements:      number of macro elements
            %   nSCells:        number of cells in S-grid
            
            %Initialize sigma_c
            self.Sigma_c = 1e0*eye(self.gridRF.nCells);
            
            nSX = numel(self.fineGridX); nSY = numel(self.fineGridY);
            if any(self.interpolationMode)
                nSX = nSX + 1; nSY = nSY + 1;
            end
            self.sigma_cf.s0 = ones(nSX*nSY, 1);  %variance field of p_cf
            
            %Initialize hyperparameters
            if strcmp(self.prior_theta_c, 'RVM')
                self.gamma = 1e-4*ones(size(self.theta_c));
            elseif strcmp(self.prior_theta_c, 'VRVM')
                self.gamma = 1e-6*ones(size(self.theta_c));
            elseif strcmp(self.prior_theta_c, 'sharedVRVM')
                self.gamma = 1e-2*ones(size(self.theta_c));
            elseif strcmp(self.prior_theta_c, 'none')
                self.gamma = NaN;
            else
                error('What prior model for theta_c?')
            end
            
            %Initialize parameters of variational approximate distributions
            self.variational_mu{1} = -8.5*ones(1, self.gridRF.nCells);
            self.variational_mu = repmat(self.variational_mu, nData, 1);
            
            self.variational_sigma{1} = 1e-4*ones(1, self.gridRF.nCells);
            self.variational_sigma = repmat(self.variational_sigma, nData, 1);
            
            %Bring variational distribution params in form for
            %unconstrained optimization
            varDistParamsVecInit{1} = [self.variational_mu{1},...
                -2*log(self.variational_sigma{1})];
            self.varDistParamsVec = repmat(varDistParamsVecInit, nData, 1);
        end
        
        function splitRFcells(self, splt_cells)
            %Split cell of random field and initialize corresponding params
            %from coarser random field discretization
            
            nElc = size(self.Sigma_c, 1);
            self.splitted_cells = [self.splitted_cells, splt_cells];
            for splt_cll = self.splitted_cells
                if isnan(self.cell_dictionary(splt_cll))
                    warning('Trying to split an already splitted cell. Skip.')
                else
                    self.gridRF.split_cell(self.gridRF.cells{splt_cll});
                    
                    %extend Sigma_c
                    index = self.cell_dictionary(splt_cll);
                    self.Sigma_c = blkdiag(self.Sigma_c,...
                        self.Sigma_c(index, index)*eye(4));
                    self.Sigma_c(index, :) = [];
                    self.Sigma_c(:, index) = [];
                    
                    %extend theta_c
                    if(~isempty(self.theta_c) && strcmp(self.mode, 'local'))
                        theta_c_mat = reshape(self.theta_c, [], nElc);
                        theta_c = theta_c_mat;
                        theta_c(:, index) = [];
                        theta_c = theta_c(:);
                        self.theta_c=[theta_c;repmat(theta_c_mat(:,index),4,1)];
                    end
                    
                    %extend hyperparameter gamma
                    if(~isempty(self.gamma) && strcmp(self.mode, 'local'))
                        gamma_mat = reshape(self.gamma, [], nElc);
                        gamma = gamma_mat;
                        gamma(:, index) = [];
                        gamma = gamma(:);
                        self.gamma = [gamma; repmat(gamma_mat(:, index), 4, 1)];
                    end
                    
                    %extend Sigma_theta_c
                    if(~isempty(self.Sigma_theta_c)&& strcmp(self.mode,'local'))
                        nFeatures = size(self.Sigma_theta_c, 1)/nElc;
                        Sigma_theta_c_k = self.Sigma_theta_c(...
                            ((index - 1)*nFeatures + 1):(index*nFeatures),...
                            ((index - 1)*nFeatures + 1):(index*nFeatures));
                        self.Sigma_theta_c(((index - 1)*nFeatures + 1):...
                            (index*nFeatures), :) = [];
                        self.Sigma_theta_c(:, ((index - 1)*nFeatures + 1):...
                            (index*nFeatures)) = [];
                        self.Sigma_theta_c = blkdiag(self.Sigma_theta_c, ...
                            Sigma_theta_c_k, Sigma_theta_c_k, ...
                            Sigma_theta_c_k, Sigma_theta_c_k);
                    end
                    
                    %extend variational parameters
                    for n = 1:numel(self.variational_mu)
                        mu_k = self.variational_mu{n}(index);
                        sigma_k = self.variational_sigma{n}(index);
                        self.variational_mu{n}(index) = [];
                        self.variational_sigma{n}(index) = [];
                        self.variational_mu{n} =...
                            [self.variational_mu{n}, mu_k*ones(1, 4)];
                        self.variational_sigma{n} = ...
                            [self.variational_sigma{n}, sigma_k*ones(1,4)];
                        self.varDistParamsVec{n} = [self.variational_mu{n},...
                            -2*log(self.variational_sigma{n})];
                    end
                    
                    %Update cell index dictionary
                    self.cell_dictionary(splt_cll) = nan;
                    self.cell_dictionary((splt_cll + 1):end) = ...
                        self.cell_dictionary((splt_cll + 1):end) - 1;
                    if isnan(self.cell_dictionary(end))
                        self.cell_dictionary = [self.cell_dictionary, ...
                            (self.cell_dictionary(end - 1) + 1):...
                            (self.cell_dictionary(end - 1) + 4)];
                    else
                        self.cell_dictionary = [self.cell_dictionary, ...
                            (self.cell_dictionary(end) + 1):...
                            (self.cell_dictionary(end) + 4)];
                    end
                    %cll_dict = self.cell_dictionary
                end
            end
            self.rf2fem = self.gridRF.map2fine(self.coarseGridX,...
                self.coarseGridY);
        end
        
        %depreceated
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

            addpath('./mesh');
            load('./data/gridRF.mat');
            self.gridRF = gridRF;
            self.rf2fem = gridRF.map2fine(coarseMesh.gridX, coarseMesh.gridY);
            
            self.sigma_cf.s0 = dlmread('./data/sigma_cf')';
            load('./data/gridS.mat');
            self.fineGridX = fineGridX;
            self.fineGridY = fineGridY;
            
            load('./data/interpolationMode.mat');
            self.interpolationMode = interpolationMode;
            
            load('./data/smoothingParameter.mat');
            self.smoothingParameter = smoothingParameter;
            load('./data/boundarySmoothingPixels.mat');
            self.boundarySmoothingPixels = boundarySmoothingPixels;
            
            load('./data/boundarySmoothingPixels.mat');
            self.boundarySmoothingPixels = boundarySmoothingPixels;
            
            try
                load('./data/Sigma_theta_c.mat');
                self.Sigma_theta_c = Sigma_theta_c;
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
        
        function printCurrentParams(self)
            %Print current model params on screen
            
            if strcmp(self.mode, 'local')
                disp('theta_c: row = feature, column = macro-cell:')
                curr_theta_c = reshape(self.theta_c,...
                    numel(self.theta_c)/self.gridRF.nCells,...
                    self.gridRF.nCells)
                curr_Sigma_c = full(diag(self.Sigma_c))
                if strcmp(self.prior_theta_c, 'sharedVRVM')
                    curr_gamma = self.gamma(1:...
                        (numel(self.theta_c)/self.gridRF.nCells));
                else
                    curr_gamma = self.gamma;
                end
            else
                curr_theta_c = self.theta_c
                curr_Sigma_c = full(diag(self.Sigma_c))
                curr_gamma = self.gamma;
            end
            curr_gamma = [curr_gamma(:), (1:numel(curr_gamma))']
            
        end
        
        function fineScaleInterp(self, X)
            %Precompute shape function interp. on every fine scale vertex
            
            nData = numel(X);
            for n = 1:nData
                self.W_cf{n} = shapeInterp(self.coarseMesh, X{n});
            end
        end
        
        function plot_params(self, figHandle, iter)
            %Plots the current theta_c
            
            %short notation
            nSX = numel(self.fineGridX); nSY = numel(self.fineGridY);
            if any(self.interpolationMode)
                nSX = nSX + 1; nSY = nSY + 1;
            end
            
            if nargin < 2
                figHandle = figure;
            end
            
            sb1 = subplot(3, 2, 1, 'Parent', figHandle);
            if(isempty(self.p_theta) ||...
                    numel(self.theta_c) ~= numel(self.p_theta))
                %random colors
                colors = [1 0 0; 0 1 0; 0 0 1; 1 0 1; 0 1 1; 0 0 0];
                for d = 1:numel(self.theta_c)
                    self.p_theta{d} = animatedline('color',...
                        colors(mod(d, 6) + 1, :), 'Parent', sb1);
                end
                sb1.XLabel.String = 'iter';
                sb1.YLabel.String = '$\theta_c$';
            end
            for d = 1:numel(self.theta_c)
                addpoints(self.p_theta{d}, iter, self.theta_c(d));
            end
            axis(sb1, 'tight');
            axis(sb1, 'fill');
            
            sb2 = subplot(3, 2, 2, 'Parent', figHandle);
            bar(self.theta_c, 'linewidth', 1, 'Parent', sb2)
            axis(sb2, 'tight');
            axis(sb2, 'fill');
            sb2.XLabel.String = 'component $i$';
            sb2.YLabel.String = '$\theta_{c,i}$';
            
            sb3 = subplot(3, 2, 3, 'Parent', figHandle);
            if(isempty(self.p_sigma) || ... 
                    numel(self.p_sigma) ~= self.gridRF.nCells)
                %random colors
                for d = 1:self.gridRF.nCells
                    self.p_sigma{d} = animatedline('color',...
                        colors(mod(d, 6) + 1, :), 'Parent', sb3);
                end
                sb3.XLabel.String = 'iter';
                sb3.YLabel.String = '$\sigma_k$';
                sb3.YScale = 'log';
            end
            for d = 1:self.gridRF.nCells
                addpoints(self.p_sigma{d}, iter, self.Sigma_c(d, d));
            end
            axis(sb3, 'tight');
            axis(sb3, 'fill');
            
            sb4 = subplot(3, 2, 4, 'Parent', figHandle);
            
            sigma_c_plot = sqrt(self.rf2fem*diag(self.Sigma_c));
            im = imagesc(reshape(sigma_c_plot, self.coarseMesh.nElX,...
                self.coarseMesh.nElY)', 'Parent', sb4);
            sb4.YDir = 'normal';
            sb4.Title.String = '$\sigma_k$';
            colorbar('Parent', figHandle);
            sb4.GridLineStyle = 'none';
            axis(sb4, 'square');
            
            sb5 = subplot(3, 2, 5, 'Parent', figHandle);
            if strcmp(self.prior_theta_c, 'sharedVRVM')
                gam = self.gamma(1:(numel(self.theta_c)/self.gridRF.nCells));
            else
                gam = self.gamma;
            end
            if isempty(self.p_gamma)
                %random colors
                for d = 1:numel(gam)
                    self.p_gamma{d} = animatedline('color',...
                        colors(mod(d, 6) + 1, :), 'Parent', sb5);
                end
                sb3.XLabel.String = 'iter';
                sb5.YLabel.String = '$\gamma$';
                sb5.YScale = 'log';
            end
            for d = 1:numel(gam)
                addpoints(self.p_gamma{d}, iter, self.gamma(d));
            end
            axis(sb5, 'tight');
            axis(sb5, 'fill');
            
            sb6 = subplot(3, 2, 6, 'Parent', figHandle);
            imagesc(reshape(sqrt(self.sigma_cf.s0), nSX, nSY)', 'Parent', sb6)
            sb6.Title.String = 'S';
            colorbar('Parent', figHandle);
            sb6.GridLineStyle = 'none';
            axis(sb6, 'square');
            sb6.YDir = 'normal';
            drawnow;
        end
                
        function write2file(self, params)
            if ~exist('./data/', 'dir')
                mkdir('./data/');
            end
            
            if contains(params, 'elbo')
                filename = './data/elbo';
                elbo = self.elbo;
                save(filename, 'elbo', '-ascii', '-append');
            end
            
            if contains(params, 'cell_score')
                filename = './data/cell_score';
                cell_score = self.cell_score';
                save(filename, 'cell_score', '-ascii', '-append');
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

            %gamma
            if contains(params, 'thetaPriorHyperparam')
                filename = './data/thetaPriorHyperparam';
                thetaPriorHyperparam = self.gamma';
                save(filename, 'thetaPriorHyperparam', '-ascii', '-append');
            end
            
            %theta_c
            if contains(params, 'theta_c')
                filename = './data/theta_c';
                tc = self.theta_c';
                save(filename, 'tc', '-ascii', '-append');
            end
            
            %Sigma_theta_c (variance of posterior on theta_c)
            if contains(params, 'sigma_theta_c')
                filename = './data/Sigma_theta_c.mat';
                Sigma_theta_c = self.Sigma_theta_c;
                save(filename, 'Sigma_theta_c');
            end
                        
            %sigma
            if contains(params, 'sigma_c')
                filename = './data/sigma_c';
                sc = full(diag(self.Sigma_c))';
                save(filename, 'sc', '-ascii', '-append');
            end
            
            %S
            if contains(params, 'sigma_cf')
                filename = './data/sigma_cf';
                scf = self.sigma_cf.s0';
                onlyFinal = true;
                if onlyFinal
                    save(filename, 'scf', '-ascii');
                else
                    save(filename, 'scf', '-ascii', '-append');
                end
            end
            
            %Parameters of variational distributions on transformed lambda_c
            if contains(params, 'vardist')
                varmu = self.variational_mu;
                varsigma = self.variational_sigma;
                save('./data/vardistparams.mat', 'varmu', 'varsigma');
            end
            
        end
        
        function set_summation_matrix(self, X)
            %sets up sum_in_macrocell matrix
            nx = numel(self.fineGridX) + 1;
            ny = numel(self.fineGridY) + 1;
            self.sum_in_macrocell = zeros(self.gridRF.nCells, nx*ny);
            
            kk = 1;
            for k = 1:numel(self.gridRF.cells)
                if isvalid(self.gridRF.cells{k})
                    self.sum_in_macrocell(kk, :) =...
                        self.gridRF.cells{k}.inside(X)';
                    kk = kk + 1;
                end
            end
        end
        
        function compute_elbo(self, N, XMean, XSqMean, X_vtx, EMiter, fig)
            %General form of elbo allowing model comparison
            %   N:                   number of training samples
            %   XMean, XSqMean:      first and second moments of transformed
            %                        lambda_c
            
            assert(~isempty(self.interpolationMode),...
                'Elbo only implemented with fixed dim(U_f)')
            %ONLY VALID IF QoI IS PRESSURE ONLY
            %Short hand notation
            N_dof = numel(self.fineGridX)*numel(self.fineGridY);
            D_c = self.gridRF.nCells;
            aa = self.VRVM_a;
            bb = self.VRVM_b;
            cc = self.VRVM_c;
            dd = self.VRVM_d;
            ee = self.VRVM_e;
            ff = self.VRVM_f;
            D_theta_c = numel(self.theta_c);
            if strcmp(self.prior_theta_c, 'sharedVRVM')
                D_gamma = D_theta_c/D_c; %for shared RVM only!
            else
                D_gamma = D_theta_c;
            end
            
            Sigma_lambda_c = XSqMean - XMean.^2;
            %sum over N and macro-cells
            sum_logdet_lambda_c = sum(sum(log(Sigma_lambda_c)));

            try
                if strcmp(self.prior_theta_c, 'sharedVRVM')
                    logdet_Sigma_theta_ck = zeros(D_c, 1);
                    for k = 1:D_c
                        logdet_Sigma_theta_ck(k) = logdet(self.Sigma_theta_c(...
                            ((k-1)*D_gamma + 1):(k*D_gamma),...
                            ((k - 1)*D_gamma + 1):(k*D_gamma)), 'chol');
                    end
                    logdet_Sigma_theta_c = sum(logdet_Sigma_theta_ck);
                else
                    logdet_Sigma_theta_c = logdet(self.Sigma_theta_c, 'chol');
                end
            catch
                logdet_Sigma_theta_c = logdet(self.Sigma_theta_c);
                warning('Sigma_theta_c not pos. def.')
            end
            
            self.elbo = -.5*N*N_dof*log(2*pi) +.5*sum_logdet_lambda_c + ...
                .5*N*D_c + N_dof*(ee*log(ff) + log(gamma(self.e)) -...
                log(gamma(ee))) - self.e*sum(log(self.f)) + D_c*(cc*log(dd) +...
                log(gamma(self.c)) - log(gamma(cc))) -...
                self.c*sum(log(self.d)) + D_gamma*(aa*log(bb) +...
                log(gamma(self.a)) - log(gamma(aa))) - ...
                self.a*sum(log(self.b(1:D_gamma))) + ...
                .5*logdet_Sigma_theta_c + .5*D_theta_c;
            if strcmp(self.prior_theta_c, 'sharedVRVM')
                gamma_expected = psi(self.a) - log(self.b);
                self.elbo = self.elbo + (D_c - 1)*sum(.5*gamma_expected -...
                    (self.a./self.b).*(self.b - bb));
            end
            self.set_summation_matrix(X_vtx);
            self.cell_score = .5*sum(log(Sigma_lambda_c), 2) - ...
                self.c*log(self.d) + .5*logdet_Sigma_theta_ck;
            f_contribution = - self.e*log(self.f);

            self.cell_score_full = self.cell_score +...
                self.sum_in_macrocell*f_contribution;
            
            sp1 = subplot(3, 4, 1, 'Parent', fig);
            hold(sp1, 'on')
            sp1.Title.String = '$\sum \log \det \lambda_c$';
            plot(EMiter, sum_logdet_lambda_c, 'kx', 'Parent', sp1);
            axis(sp1, 'tight');
            
            sp2 = subplot(3, 4, 2, 'Parent', fig);
            hold(sp2, 'on')
            sp2.Title.String = '$\log \Gamma(e)$';
            plot(EMiter, log(gamma(self.e)), 'kx', 'Parent', sp2);
            axis(sp2, 'tight');
            
            sp3 = subplot(3, 4, 3, 'Parent', fig);
            hold(sp3, 'on')
            sp3.Title.String = '$-e \cdot \sum \log f$';
            plot(EMiter, - self.e*sum(log(self.f)), 'kx', 'Parent', sp3);
            axis(sp3, 'tight');
            
            sp4 = subplot(3, 4, 4, 'Parent', fig);
            hold(sp4, 'on')
            sp4.Title.String = '$D_c \log \Gamma(c)$';
            plot(EMiter, D_c*log(gamma(self.c)), 'kx', 'Parent', sp4);
            axis(sp4, 'tight');
            
            sp5 = subplot(3, 4, 5, 'Parent', fig);
            hold(sp5, 'on')
            sp5.Title.String = '$c\sum \log(d)$';
            plot(EMiter, -self.c*sum(log(self.d)), 'kx', 'Parent', sp5);
            axis(sp5, 'tight');
            
            sp6 = subplot(3, 4, 6, 'Parent', fig);
            hold(sp6, 'on')
            sp6.Title.String = '$D_\gamma \cdot \log(\Gamma(a))$';
            plot(EMiter, D_gamma*log(gamma(self.a)), 'kx', 'Parent', sp6);
            axis(sp6, 'tight');
            
            sp7 = subplot(3, 4, 7, 'Parent', fig);
            hold(sp7, 'on')
            sp7.Title.String = '$a\sum \log b_{1:D_{\gamma}}$';
            plot(EMiter,-self.a*sum(log(self.b(1:D_gamma))),'kx','Parent', sp7);
            axis(sp7, 'tight');
            
            sp8 = subplot(3, 4, 8, 'Parent', fig);
            hold(sp8, 'on')
            sp8.Title.String = '$\frac{1}{2}\log |\Sigma_{\theta_c}|$';
            plot(EMiter, .5*logdet_Sigma_theta_c, 'kx', 'Parent', sp8);
            axis(sp8, 'tight');
            
            sp9 = subplot(3, 4, 9, 'Parent', fig);
            hold(sp9, 'on')
            sp9.Title.String = '$\frac{1}{2}(D_c - 1)\sum <\gamma>$';
            plot(EMiter, (D_c - 1)*.5*sum(gamma_expected), 'kx', 'Parent', sp9);
            axis(sp9, 'tight');
            
            sp10 = subplot(3, 4, 10, 'Parent', fig);
            hold(sp10, 'on')
            sp10.Title.String = '$-(D_c - 1)\sum \frac{a}{b}(b - b_0)$';
            plot(EMiter, -(D_c - 1)*sum((self.a./self.b).*(self.b - bb)),...
                'kx', 'Parent', sp10);
            axis(sp10, 'tight');
        end
        
        function plotElbo(self, fig, EMiter)
            sp = subplot(1, 1, 1, 'Parent', fig);
            hold(sp, 'on');
            if isempty(self.p_elbo)
                self.p_elbo = animatedline('Parent', sp);
                self.p_elbo.LineWidth = 2;
                self.p_elbo.Marker = 'x';
                self.p_elbo.MarkerSize = 10;
                sp.XLabel.String = 'iteration';
                sp.YLabel.String = 'elbo';
            end
            addpoints(self.p_elbo, EMiter, self.elbo);
            axis(sp, 'tight');
            axis(sp, 'fill');
            drawnow;
        end
    end
end

