function [L] = matrixLinealPath(diskCenters, diskRadii,...
    gridX, gridY, distance)
%Gives the lineal path function for the matrix phase according to the
%approximation in Torquato eq. 6.37


cumsumX = cumsum(gridX);
cumsumY = cumsum(gridY);

Nx = numel(gridX);
Ny = numel(gridY);

A0 = zeros(Nx, Ny);
for ny = 1:Ny
    if ny > 1
        ly = cumsumY(ny) - cumsumY(ny - 1);
    else
        ly = cumsumY(1);
    end
    for nx = 1:Nx
        if nx > 1
            lx = cumsumX(nx) - cumsumX(nx - 1);
        else
            lx = cumsumX(1);
        end
        
        %Surface of macro-element without inclusions
        A0(nx, ny) = lx*ly;
    end
end

Ncirc = length(diskRadii);
A = A0;
meanRadii = zeros(Nx, Ny);
meanSqRadii = zeros(Nx, Ny);
nRadii = zeros(Nx, Ny);
for circ = 1:Ncirc
    nx = 1;
    while(diskCenters(circ, 1) > cumsumX(nx) && nx < Nx)
        nx = nx + 1;
    end
    ny = 1;
    while(diskCenters(circ, 2) > cumsumY(ny) && ny < Ny)
        ny = ny + 1;
    end

    %THIS IS INCORRECT BUT AN APPROXIMATION!!!
    %we substract the full circle from the macro-cell the center belongs to
    %this ignores the fact that a circle may lie on multiple macro-cells
    %this should be corrected at some point
    A(nx, ny) = A(nx, ny) - pi*diskRadii(circ)^2;
    nRadii(nx, ny) = nRadii(nx, ny) + 1;
    meanRadii(nx, ny) =...
        ((nRadii(nx, ny) - 1)/(nRadii(nx, ny)))*meanRadii(nx, ny) + ...
        (1/nRadii(nx, ny))*diskRadii(circ);
    meanSqRadii(nx, ny) =...
        ((nRadii(nx, ny) - 1)/(nRadii(nx, ny)))*meanSqRadii(nx, ny) + ...
        (1/nRadii(nx, ny))*diskRadii(circ)^2;
end

porefrac = A./A0;

L = porefrac.*...
    exp(-(2*distance*(1 - porefrac).*meanRadii)./(pi*porefrac.*meanSqRadii));
L(isnan(L)) = 1; %Happens if there are no exclusions in macro-cell



end

