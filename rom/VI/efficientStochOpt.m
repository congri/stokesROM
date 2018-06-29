function [varDistParams, x] =...
    efficientStochOpt(x, log_emp_dist, variationalDist, stepWidth, dim)
%Memory efficient stocastic optimization for parfor loop
%Perform stochastic maximization step

debug = false;   %debug mode

updateRule = 'adam';
% beta1 = .7;                     %the higher, the more important is momentum
% beta2 = .8;                    %curvature parameter
beta1 = .9;
beta2 = .99999;
epsilon = 1e-8;                  %curvature stabilization parameter

stepOffset = 100;                %Robbins-Monro step offset
nSamples = 1000;                  %gradient samples per iteration
maxIterations = 1e4;
maxCompTime = 15;

converged = false;
steps = 0;

stepWidth_stepOffset = stepWidth*stepOffset;
if strcmp(variationalDist, 'diagonalGauss')
    varDistParams.mu = x(1:dim);
    varDistParams.sigma = exp(-.5*x((dim + 1):end));
elseif strcmp(variationalDist, 'fullRankGauss')
    varDistParams.mu = x(1:dim);
    %lower triangular cholesky factor L
    varDistParams.L = reshape(x((dim + 1):end), dim, dim);
    varDistParams.LT = varDistParams.L';
    varDistParams.LInv = inv(varDistParams.L);
    varDistParams.Sigma = varDistParams.L*varDistParams.LT;
else
    error('Unknown variational distribution')
end

if debug
    disp('starting stochastic optimization')
    x_0 = x
    f = figure;
    f.Units = 'normalized';
    f.OuterPosition = [0 0 1 1];
    clf(f)
    sb1 = subplot(2, 2, 1, 'Parent', f);
    hold on;
    title('$\mu$');
    sb2 = subplot(2, 2, 2, 'Parent', f);
    sb2.YScale = 'log';
    hold on;
    title('$-2\log \sigma$')
    sb3 = subplot(2, 2, 3, 'Parent', f);
    sb3.YScale = 'log';
    hold on;
    title('norm momentum');
    sb4 = subplot(2, 2, 4, 'Parent', f);
    sb4.YScale = 'log';
    hold on;
    title('norm gradient');
    
    for d = 1:dim
        p_mu(d) = animatedline('color', 'b', 'Parent', sb1);
        %semilogy(iter, varDistParams.sigma, 'rx', 'Parent', sb2);
        p_sigma(d) = animatedline('color', 'r', 'Parent', sb2);
    end
    p_momentum = animatedline('Parent', sb3);
    p_grad = animatedline('Parent', sb4);
end

tic;
while ~converged
    
    gradient =...
        sampleELBOgrad(log_emp_dist, variationalDist, nSamples, varDistParams);

    if strcmp(updateRule, 'adam')
        
        if steps == 0
            %careful first iteration
            momentum = 1e-6*gradient;
            uncenteredXVariance = gradient.^2;
        else
            momentum = beta1*momentum + (1 - beta1)*gradient;
        end
        uncenteredXVariance = beta2*uncenteredXVariance...
            + (1 - beta2)*gradient.^2;
        
        %Optimization update
        x = x + (stepWidth_stepOffset/(stepOffset + steps)).*...
            (1./(sqrt(uncenteredXVariance) + epsilon)).*momentum;
        
    elseif strcmp(updateRule, 'robbinsMonro')
        delta = ((stepWidth_stepOffset)/(stepOffset + steps)).*gradient;
        nDelta = norm(delta);
        stabilityFactor = 2;
        if(nDelta > stabilityFactor*norm(x))
            delta = (stabilityFactor/nDelta)*delta;
        end
        x = x + delta;
    else
        error('Unknown update heuristic for stochastic optimization')
    end
    steps = steps + 1;
    
    if strcmp(variationalDist, 'diagonalGauss')
        varDistParams.mu = x(1:dim);
        varDistParams.sigma = exp(-.5*x((dim + 1):end));
    elseif strcmp(variationalDist, 'fullRankGauss')
        varDistParams.mu = x(1:dim);
        %lower triangular cholesky factor L
        varDistParams.L = reshape(x((dim + 1):end), dim, dim);
        varDistParams.LT = varDistParams.L';
        varDistParams.LInv = inv(varDistParams.L);
        varDistParams.Sigma = varDistParams.L*varDistParams.LT;
    else
        error('Unknown variational distribution')
    end
    
    if debug
        plotStep = 10;
        if(mod(steps, plotStep) == 0 && steps > plotStep)
            %mu = varDistParams.mu
            %sigma = varDistParams.sigma
%             plot(iter, varDistParams.mu, 'bx', 'Parent', sb1);
% %             semilogy(iter, varDistParams.sigma, 'rx', 'Parent', sb2);
%             semilogy(iter, x((dim + 1):end), 'rx', 'Parent', sb2);
%             semilogy(iter, norm(momentum), 'kx', 'Parent', sb3);
%             semilogy(iter, norm(gradient), 'kx', 'Parent', sb4);
            addpoints(p_grad, steps, norm(gradient));
            addpoints(p_momentum, steps, norm(momentum));
            for d = 1:dim
                addpoints(p_mu(d), steps, varDistParams.mu(d));
                addpoints(p_sigma(d), steps, varDistParams.sigma(d));
            end
            drawnow
        end
    end
    
    compTime = toc;
    if steps > maxIterations
        converged = true;
        disp('Converged because max number of iterations exceeded')
    elseif compTime > maxCompTime
        converged = true;
        disp('Converged because max computation time exceeded')
    end
    if mod(steps, 100000) == 0
        nSamples = nSamples + 1;%this is purely heuristic! remove if unwanted
    end
end

if strcmp(variationalDist, 'diagonalGauss')
    varDistParams.XSqMean = varDistParams.sigma.^2 + varDistParams.mu.^2;
elseif strcmp(variationalDist, 'fullRankGauss')
    varDistParams.XSqMean =...
        diag(varDistParams.Sigma + varDistParams.mu'*varDistParams.mu);
else
    error('Unknown variational distribution')
end


