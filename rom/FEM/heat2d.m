function [Out] = heat2d(mesh, D)
%2D heat conduction main function
%Gives back temperature on point x

%get_loc_stiff as nested function for performance
Dmat = spalloc(8, 8, 16);

    function [diffusionStiffness] = get_loc_stiff2(Bvec, D)
        %Gives the local stiffness matrix
        
        Dmat(1:2, 1:2) = D;
        Dmat(3:4, 3:4) = D;
        Dmat(5:6, 5:6) = D;
        Dmat(7:8, 7:8) = D;
        
        diffusionStiffness = Bvec'*Dmat*Bvec;
    end


%Compute local stiffness matrices, once and for all
Out.diffusionStiffness = zeros(4, 4, mesh.nEl);

for e = 1:mesh.nEl
    Out.diffusionStiffness(:, :, e) = get_loc_stiff2(mesh.Bvec(:, :, e),...
        D(:, :, e));
end

%Global stiffness matrix
localStiffness = Out.diffusionStiffness;

Out.globalStiffness = get_glob_stiff2(mesh, localStiffness);
%Global force vector
Out.globalForce = get_glob_force(mesh, localStiffness);

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