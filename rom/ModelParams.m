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
        prior_theta_c = 'none'
        gamma   %Gaussian precision of prior on theta_c
        
        %% Parameters of variational distributions
        variational_mu
        variational_sigma
    end
    
    methods
        function self = ModelParams()
            %Constructor
        end
        
        function self = initialize(self, nFeatures, nElements, nData)
            %Initialize model parameters
            %   nFeatures:      number of feature functions
            %   nElements:      number of macro elements
            
            %Initialize theta_c to 0
            self.theta_c = zeros(nFeatures, 1);
            %Initialize sigma_c to I
            self.Sigma_c = eye(nElements);
                        
            self.sigma_cf.type = 'delta';
            self.sigma_cf.s0 = @(x) 1;  %standard deviation field of p_cf
            
            %Initialize hyperparameters
            if strcmp(self.prior_theta_c, 'RVM')
                self.gamma = ones(size(self.theta_c));
            elseif strcmp(self.prior_theta_c, 'none')
                self.gamma = [];
            else
                error('What prior model for theta_c?')
            end
            
            %Initialize parameters of variational approximate distributions
            self.variational_mu{1} = zeros(1, nElements);
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
                thetaArray, SigmaArray, coarseMesh)
            %Plots the current theta_c
            
           figure(figHandle);
            sb = subplot(3, 2, 1, 'Parent', figHandle);
            plot(thetaArray, 'linewidth', 1, 'Parent', sb)
            axis tight;
            ylim([(min(thetaArray(end, :)) - 1) (max(thetaArray(end, :)) + 1)]);
            sb = subplot(3,2,2, 'Parent', figHandle);
            bar(self.theta_c, 'linewidth', 1, 'Parent', sb)
            axis tight;
            sb = subplot(3,2,3, 'Parent', figHandle);
            semilogy(sqrt(SigmaArray), 'linewidth', 1, 'Parent', sb)
            axis tight;
            sb = subplot(3,2,4, 'Parent', figHandle);
            imagesc(reshape(diag(sqrt(self.Sigma_c(1:coarseMesh.nEl,...
                1:coarseMesh.nEl))),...
                coarseMesh.nElX, coarseMesh.nElY), 'Parent', sb)
            title('\sigma_k')
            colorbar
            grid off;
            axis tight;
            sb = subplot(3,2,5, 'Parent', figHandle);
            bar(self.gamma, 'linewidth', 1, 'Parent', sb)
            axis tight;
            drawnow
        end
    end
end

