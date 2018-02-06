function [lc] = meanChordLength(diskCenters, diskRadii, gridX, gridY)
%Mean chord length for non-overlapping polydisp. spheres
%according to Torquato 6.40


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
meanRadii = zeros(Nx, Ny);
meanSqRadii = zeros(Nx, Ny);
nRadii = zeros(Nx, Ny);
A = A0;
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
    nRadii(nx, ny) = nRadii(nx, ny) + 1;
    meanRadii(nx, ny) =...
        ((nRadii(nx, ny) - 1)/(nRadii(nx, ny)))*meanRadii(nx, ny) + ...
        (1/nRadii(nx, ny))*diskRadii(circ);
    meanSqRadii(nx, ny) =...
        ((nRadii(nx, ny) - 1)/(nRadii(nx, ny)))*meanSqRadii(nx, ny) + ...
        (1/nRadii(nx, ny))*diskRadii(circ)^2;
    A(nx, ny) = A(nx, ny) - pi*diskRadii(circ)^2;
end
porefrac = A./A0;
exclfrac = 1 - porefrac;



lc = .5*pi*(meanSqRadii./meanRadii).*(porefrac./exclfrac);
if any(any(meanRadii == 0))
    %warning('Setting mean chord length of cells with no inclusion to 0.')
    lc(isnan(lc)) = sqrt(A(isnan(lc))); %set to ~ cell edge length
end
