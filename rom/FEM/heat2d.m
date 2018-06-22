function [Out] = heat2d(mesh, D)
%2D heat conduction main function
%Gives back temperature on point x

%get_loc_stiff as nested function for performance
% Dmat = spalloc(8, 8, 16);
row = [1 1 2 2 3 3 4 4 5 5 6 6 7 7 8 8]';
col = [1 2 1 2 3 4 3 4 5 6 5 6 7 8 7 8]';

    function [diffusionStiffness] = get_loc_stiff2(Bvec, D)
        %Gives the local stiffness matrix
        
%         Dmat(1:2, 1:2) = D;
%         Dmat(3:4, 3:4) = D;
%         Dmat(5:6, 5:6) = D;
%         Dmat(7:8, 7:8) = D;
        Dmat = sparse(row, col, repmat(D(:), 4, 1));
        
        diffusionStiffness = Bvec'*Dmat*Bvec;
    end


%Compute local stiffness matrices, once and for all
Out.diffusionStiffness = zeros(4, 4, mesh.nEl);

isotropicDiffusivity = true;
if isotropicDiffusivity
    for e = 1:mesh.nEl
        Out.diffusionStiffness(:, :, e) = D(e)*mesh.d_loc_stiff(:, :, e);
    end
else
    for e = 1:mesh.nEl
        Out.diffusionStiffness(:, :, e) = get_loc_stiff2(mesh.Bvec(:, :, e),...
            D(:, :, e));
    end
end

% %Global stiffness matrix
% localStiffness = Out.diffusionStiffness;

Out.globalStiffness = get_glob_stiff2(mesh, Out.diffusionStiffness);
%Global force vector
Out.globalForce = get_glob_force(mesh, Out.diffusionStiffness);

%Finally solving the equation system
Out.naturalTemperatures = Out.globalStiffness\Out.globalForce;


%Temperature field
Tf = zeros(mesh.nNodes, 1);
Tf(mesh.id) = Out.naturalTemperatures;
Tff = zeros(mesh.nElX + 1, mesh.nElY + 1);

for i = 1:mesh.nNodes
    Tff(i) = Tf(i);
    if(any(i == mesh.essentialNodes))
        %node i is essential
        Tff(i) = mesh.essentialTemperatures(i);
    end
end

Tff = Tff';
Out.Tff = Tff;


end