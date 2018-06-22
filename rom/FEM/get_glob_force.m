function [F] = get_glob_force(mesh, k)
%Assemble global force vector

neq = max(mesh.nodalCoordinates(3, :));
F = zeros(neq, 1);
Tbflag = false;
Tb = zeros(4, 1);
for e = 1:mesh.nEl
%     f = get_loc_force(e, domain, k);
    %Contribution due to essential boundaries
    %local stiffness matrix k
    
    %Boundary value temperature of element e
    if Tbflag
        Tb = zeros(4, 1);
    end
    Tbflag = false;
    for i = 1:4
        if(any(mesh.globalNodeNumber(e, i) == mesh.essentialNodes))
            Tb(i) = mesh.essentialTemperatures(mesh.globalNodeNumber(e, i));
            Tbflag = true;
        end
    end


    for ln = 1:4
        eqn = mesh.lm(e, ln);
        if(eqn ~= 0)
            F(eqn) = F(eqn) + mesh.f_tot(ln, e);
            if Tbflag
                fT = k(:, :, e)*Tb;
                F(eqn) = F(eqn) - fT(ln);
            end
        end
    end
end

end

