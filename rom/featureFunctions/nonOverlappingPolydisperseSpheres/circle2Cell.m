function [cellOfCircle] = circle2Cell(diskCenters, gridX,...
    gridY)
%Mapping from circle exclusion to macro-cell given by grid vectors gridX,
%gridY
cumsumX = cumsum(gridX);
cumsumY = cumsum(gridY);

Nx = numel(gridX);
Ny = numel(gridY);

cellOfCircle = zeros(size(diskCenters, 1), 1);
for crcl = 1:size(diskCenters, 1)
    nx = 1;
    while(diskCenters(crcl, 1) > cumsumX(nx) && nx < Nx)
        nx = nx + 1;
    end
    
    ny = 1;
    while(diskCenters(crcl, 2) > cumsumY(ny) && ny < Ny)
        ny = ny + 1;
    end
    cellOfCircle(crcl) = nx + (ny - 1)*Nx;
end
end

