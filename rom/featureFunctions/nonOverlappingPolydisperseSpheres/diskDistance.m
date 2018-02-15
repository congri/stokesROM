function [distQuantity] = diskDistance(diskCenters, diskRadii,...
    gridRF, property, p_norm)
%Computes average/min/max/std distance of disks in a macro-cell

distQuantity = zeros(gridRF.nCells, 1);
n = 1;
for cll = gridRF.cells
    if isvalid(cll{1})
        centers = diskCenters(cll{1}.inside(diskCenters), :);
        radii = diskRadii(cll{1}.inside(diskCenters));
        distances = zeros(numel(radii)*(numel(radii) - 1)/2, 1);
        ind = 1;
        if numel(radii) > 1
            for i = 1:numel(radii)
                for j = (i + 1):numel(radii)
                    if strcmp(p_norm, 'edge2edge')
                        %Computes 2-norm, but from disk edge to disk edge
                        distances(ind) = norm(centers(i, :) - centers(j, :));
                        distances(ind) = distances(ind) - radii(i) - radii(j);
                    else
                        %Computes p-norm between disk centers
                        distances(ind) = norm(centers(i, :) - centers(j, :),...
                            p_norm);
                    end
                    ind = ind +1;
                end
            end
        else
            %warning('Cell with one or less exclusions. Setting distances = 0')
            distances = 0;
        end
        
        if strcmp(property, 'mean')
            distQuantity(n) = mean(distances);
        elseif strcmp(property, 'max')
            distQuantity(n) = max(distances);
        elseif strcmp(property, 'min')
            distQuantity(n) = min(distances);
        elseif strcmp(property, 'std')
            distQuantity(n) = std(distances);
        elseif strcmp(property, 'var')
            distQuantity(n) = var(distances);
        else
            error('Unknown distance property')
        end
        n = n + 1;
    end
end

end

