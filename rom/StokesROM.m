classdef StokesROM
    %Class for reduced order model of Stokes equation
    
    properties
        %ROM FEM grid vectors of rectangular mesh
        gridX = .25*ones(1, 4)
        gridY = .25*ones(1, 4)
        coarseMesh      %mesh object
        
        %Grid of p_cf variance
        gridSX
        gridSY
        
        %StokesData object
        trainingData
        
        %Model parameters
        modelParams
    end
    
    methods
        function [self] = StokesROM(gridX, gridY, gridSX, gridSY, p_bc, u_bc)
            %Constructor
            self.gridX = gridX; nX = length(gridX);
            self.gridY = gridY; nY = length(gridY);
            
            self.gridSX = gridSX;
            self.gridSY = gridSY;
            
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
            
            %Coarse mesh object
            self.coarseMesh = Mesh(gridX, gridY);
            self.coarseMesh = self.coarseMesh.setBoundaries(2:(2*nX + 2*nY),...
                p_bc, u_bc_handle);
        end
        
        function [self] = readTrainingData(self, samples, u_bc)
            %Sets trainingData to StokesData object
            self.trainingData = StokesData(samples, u_bc);
            self.trainingData = self.trainingData.readData('px');
        end
        
        function [self] = initializeModelParams(self)
            %Initialize params theta_c, theta_cf
            
            if isempty(self.trainingData.designMatrix)
                self.trainingData = self.trainingData.evaluateFeatures(...
                    self.gridX, self.gridY);
            end
            
            %Initialize theta_c to 0
            self.modelParams = ModelParams;
            nFeatures = size(self.trainingData.designMatrix{1}, 2);
            nElements = numel(self.gridX)*numel(self.gridY);
            nData = numel(self.trainingData.samples);
            nSCells = numel(self.gridSX)*numel(self.gridSY);
            
            self.modelParams = self.modelParams.initialize(nFeatures,...
                nElements, nData, nSCells);
        end
        
        function self = M_step(self, XMean, XSqMean, sqDist_p_cf)
            %Update parameters in p_c
            self = self.update_p_c(XMean, XSqMean);
            self = self.update_p_cf(sqDist_p_cf);
        end
        
        function self = update_p_c(self, XMean, XSqMean)
            %% Find optimal theta_c and Sigma_c self-consistently:
            %update theta_c, then Sigma_c, then theta_c and so on
            
            %short-hand notation
            dim_theta_c = numel(self.modelParams.theta_c);
            N_train = numel(self.trainingData.samples);
            nElc = self.coarseMesh.nEl;
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
        
        function self = update_p_cf(self, sqDist_p_cf)
            
            Ncells_gridS = numel(self.gridSX)*numel(self.gridSY);
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
        
        function [] = plotCurrentState(self, fig, dataOffset, condTransOpts)
            %Plots the current modal effective property and the modal
            %reconstruction for 2 -training- samples
            for i = 1:2
                Lambda_eff_mode = conductivityBackTransform(...
                    self.trainingData.designMatrix{i + dataOffset}*...
                    self.modelParams.theta_c, condTransOpts);
                sb1 = subplot(2, 3, 1 + (i - 1)*3, 'Parent', fig);
                imagesc(reshape(Lambda_eff_mode, self.coarseMesh.nElX,...
                    self.coarseMesh.nElY)', 'Parent', sb1)
                sb1.YDir = 'normal';
                axis(sb1, 'tight');
                axis(sb1, 'square');
                sb1.GridLineStyle = 'none';
                sb1.XTick = [];
                sb1.YTick = [];
                cbp_lambda = colorbar('Parent', fig);
                sb2 = subplot(2, 3, 2 + (i - 1)*3, 'Parent', fig);
                if isempty(self.trainingData.cells)
                    self.trainingData = self.trainingData.readData('c');
                end
                trihandle = trisurf(self.trainingData.cells{i + dataOffset},...
                    self.trainingData.X{i + dataOffset}(:, 1),...
                    self.trainingData.X{i + dataOffset}(:, 2),...
                    self.trainingData.P{i + dataOffset}, 'Parent', sb2);
                trihandle.LineStyle = 'none';
                axis(sb2, 'tight');
                axis(sb2, 'square');
                sb2.View = [0, 90];
                sb2.GridLineStyle = 'none';
                sb2.XTick = [];
                sb2.YTick = [];
                sb2.Box = 'on';
                sb2.BoxStyle = 'full';
                
                cbp_true = colorbar('Parent', fig);
                
                sb3 = subplot(2, 3, 3 + (i - 1)*3, 'Parent', fig);
                D = zeros(2, 2, self.coarseMesh.nEl);
                for j = 1:self.coarseMesh.nEl
                    D(:, :, j) =  Lambda_eff_mode(j)*eye(2);
                end
                
                coarseFEMout = heat2d(self.coarseMesh, D);
                
                Tc = coarseFEMout.Tff';
                Tc = Tc(:);
                reconstruction = self.modelParams.W_cf{i + dataOffset}*Tc;
                
                trihandle2 = trisurf(self.trainingData.cells{i + dataOffset},...
                    self.trainingData.X{i + dataOffset}(:, 1),...
                    self.trainingData.X{i + dataOffset}(:, 2),...
                    reconstruction, 'Parent', sb3);
                trihandle2.LineStyle = 'none';
                trihandle2.FaceColor = 'b';
                hold(sb3, 'on');
                trihandle3 = trisurf(self.trainingData.cells{i + dataOffset},...
                    self.trainingData.X{i + dataOffset}(:, 1),...
                    self.trainingData.X{i + dataOffset}(:, 2),...
                    self.trainingData.P{i + dataOffset}, 'Parent', sb3);
                trihandle3.LineStyle = 'none';
                hold(sb3, 'off');
                axis(sb3, 'tight');
                axis(sb3, 'square');
                sb3.Box = 'on';
                sb3.BoxStyle = 'full';
                sb3.XTick = [];
                sb3.YTick = [];
                cbp_reconst = colorbar('Parent', fig);
            end
            drawnow
        end
    end
end

