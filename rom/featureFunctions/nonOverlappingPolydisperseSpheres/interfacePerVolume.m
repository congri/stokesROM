function [relativeInterfaceArea] = interfacePerVolume(...
    diskCenters, diskRadii, gridX, gridY)
%Computes the volume fraction of microstructures with disk exclusions
%   diskCenters:         clear
%   diskRadii:           clear
%   gridVectorX/Y:       specification of macro-cell edge lengths


%add small number at last element to include vertices on corresponding
%domain boundary
cumsumX = cumsum(gridX);    cumsumX(end) = cumsumX(end) + 1e-12;
cumsumY = cumsum(gridY);    cumsumY(end) = cumsumY(end) + 1e-12;

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
interfaceArea = 0*A0;          %interfaces are of dimension 1 in 2d
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
    %Assume that a disk belongs to a macro-cell if its center is in the cell
    %this ignores the fact that a circle may lie on multiple macro-cells
    %this should be corrected at some point
    interfaceArea(nx, ny) = interfaceArea(nx, ny) + 2*pi*diskRadii(circ);
end

relativeInterfaceArea = interfaceArea./A0;


