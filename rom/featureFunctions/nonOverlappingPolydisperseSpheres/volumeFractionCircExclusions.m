function [porefrac] = volumeFractionCircExclusions(...
    diskCenters, diskRadii, gridVectorX, gridVectorY)
%Computes the pore fraction of microstructures with disk exclusions
%pores == voids where fluid can flow
%   diskCenters:         clear
%   diskRadii:           clear
%   gridVectorX/Y:       specification of macro-cell edge lengths

cumsumX = cumsum(gridVectorX);
cumsumY = cumsum(gridVectorY);

Nx = numel(gridVectorX);
Ny = numel(gridVectorY);

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
end

porefrac = A./A0;

end

