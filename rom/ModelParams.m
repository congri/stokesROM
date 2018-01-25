classdef ModelParams
    %Initialize, update, ... the ROM model params
    
    properties
        %% Model parameters
        %p_c
        theta_c
        Sigma_c
        
        %p_cf
        W_cf
        sigma_cf
        
        %Surrogate FEM mesh
        coarseMesh
        
        %% Model hyperparameters
        prior_theta_c = 'RVM'
        gamma   %Gaussian precision of prior on theta_c
        
        %% Parameters of variational distributions
        variational_mu
        variational_sigma
    end
    
    methods
        function self = ModelParams()
            %Constructor
        end
        
        function self = initialize(self, nFeatures, nElements, nData, nSCells)
            %Initialize model parameters
            %   nFeatures:      number of feature functions
            %   nElements:      number of macro elements
            %   nSCells:        number of cells in S-grid
            
            %Initialize theta_c to 0
            self.theta_c = zeros(nFeatures, 1);
            %Initialize sigma_c to I
            self.Sigma_c = 1e-4*eye(nElements);
                        
            self.sigma_cf.type = 'delta';
            self.sigma_cf.s0 = ones(nSCells, 1);  %variance field of p_cf
            
            %Initialize hyperparameters
            if strcmp(self.prior_theta_c, 'RVM')
                self.gamma = 1e-4*ones(size(self.theta_c));
            elseif strcmp(self.prior_theta_c, 'none')
                self.gamma = NaN;
            else
                error('What prior model for theta_c?')
            end
            
            %Initialize parameters of variational approximate distributions
            self.variational_mu{1} = 0*ones(1, nElements);
            self.variational_mu = repmat(self.variational_mu, nData, 1);
            
            self.variational_sigma{1} = 1e0*ones(1, nElements);
            self.variational_sigma = repmat(self.variational_sigma, nData, 1);
        end
        
        function self = fineScaleInterp(self, X)
            %Precompute shape function interp. on every fine scale vertex
            
            nData = numel(X);
            for n = 1:nData
                self.W_cf{n} = shapeInterp(self.coarseMesh, X{n});
            end
            
            self.saveParams('W');
        end
        
        function self = plot_params(self, figHandle,...
                thetaArray, SigmaArray, nSX, nSY)
            %Plots the current theta_c
            
            %figure(figHandle);
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
            
            sb4 = subplot(3, 2, 4, 'Parent', figHandle);
            im = imagesc(reshape(diag(sqrt(self.Sigma_c(1:self.coarseMesh.nEl,...
                1:self.coarseMesh.nEl))),...
                self.coarseMesh.nElX, self.coarseMesh.nElY)', 'Parent', sb4);
            sb4.YDir = 'normal';
            sb4.Title.String = '$\sigma_k$';
            colorbar('Parent', figHandle);
            sb4.GridLineStyle = 'none';
            axis(sb4, 'square');
            
            sb5 = subplot(3, 2, 5, 'Parent', figHandle);
            bar(self.gamma, 'linewidth', 1, 'Parent', sb5)
            axis(sb5, 'tight');
            
            sb6 = subplot(3, 2, 6, 'Parent', figHandle);
            imagesc(reshape(sqrt(self.sigma_cf.s0), nSX, nSY)', 'Parent', sb6)
            sb6.Title.String = 'S';
            colorbar('Parent', figHandle);
            sb6.GridLineStyle = 'none';
            axis(sb6, 'square');
            sb6.YDir = 'normal';
        end
                
        function [] = saveParams(self, params)
            if ~exist('./data/', 'dir')
                mkdir('./data/');
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
            
        end
    end
end

