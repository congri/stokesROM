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
        
        function self = fineScaleInterp(self, X, coarseMesh)
            %Precompute shape function interp. on every fine scale vertex
            
            nData = numel(X);
            for n = 1:nData
                self.W_cf{n} = shapeInterp(coarseMesh, X{n});
            end
        end
        
        function self = plot_params(self, figHandle,...
                thetaArray, SigmaArray, coarseMesh, nSX, nSY)
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
            imagesc(reshape(diag(sqrt(self.Sigma_c(1:coarseMesh.nEl,...
                1:coarseMesh.nEl))),...
                coarseMesh.nElX, coarseMesh.nElY), 'Parent', sb4)
            sb4.Title.String = '\sigma_k';
            colorbar('Parent', figHandle);
            sb4.GridLineStyle = 'none';
            axis(sb4, 'square');
            
            sb5 = subplot(3, 2, 5, 'Parent', figHandle);
            bar(self.gamma, 'linewidth', 1, 'Parent', sb5)
            axis(sb5, 'tight');
            
            sb6 = subplot(3, 2, 6, 'Parent', figHandle);
            imagesc(reshape(sqrt(self.sigma_cf.s0), nSX, nSY), 'Parent', sb6)
            sb6.Title.String = 'S';
            colorbar('Parent', figHandle);
            sb6.GridLineStyle = 'none';
            axis(sb6, 'square');
        end
    end
end

