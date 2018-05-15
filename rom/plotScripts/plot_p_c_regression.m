function [] = plot_p_c_regression(lambda_c_mean, theta_c, designMatrix, features, mode, Nx, Ny)

if strcmp(mode, 'local')
    nFeatures = size(designMatrix{1}, 2)/size(designMatrix{1}, 1);
    for j = features
        fig(j) = figure;
        k = 1; %coarse element
        for y = 1:Ny
            for x = 1:Nx
                minX = Inf; maxX = -Inf; minY = Inf; maxY = -Inf;
                sb(k) = subplot(Nx, Ny, k);
                hold on;
                for n = 1:numel(designMatrix)
                    yData = lambda_c_mean(k, n) -...
                        designMatrix{n}(k, :)*theta_c + ...
                        designMatrix{n}(k, j + (k - 1)*nFeatures).*...
                        theta_c(j + (k - 1)*nFeatures);
                    if yData > maxY, maxY = yData; end
                    if yData < minY, minY = yData; end
                    xData = designMatrix{n}(k, j + (k - 1)*nFeatures);
                    plot(xData, yData, 'bx');
                    if xData > maxX, maxX = xData; end
                    if xData < minX, minX = xData; end
                end
                if minX ~=maxX, sb(k).XLim = [minX, maxX]; end
                ylabel('$\left<\lambda_c \right>$');
                xlabel('$\phi$');
                k = k + 1;
            end
        end
    end
else
    for j = features
        fig(j) = figure;
        for n = 1:numel(designMatrix)
            yData = lambda_c_mean(:, n) - designMatrix{n}*theta_c + ...
                designMatrix{n}(:, j).*theta_c(j);
            p = plot(designMatrix{n}(:, j), yData, 'x');
            hold on;
        end
    end
end

end

