function [d_r] = FEMgrad(FEMout, mesh)
%Compute derivatives of FEM equation system r = K*Y - F w.r.t. Lambda_e
%ONLY VALID FOR ISOTROPIC HEAT CONDUCTIVITY MATRIX D!!!

% (d/d Lambda_e) k^(e) = (1/Lambda_e) k^(e)     as k^(e) linear in Lambda_e
d_r = zeros(mesh.nEq, mesh.nEl);

for e = 1:mesh.nEl
    d_r(:, e) = (mesh.d_glob_stiff{e}*FEMout.naturalTemperatures -...
        mesh.d_glob_force{e});
end
d_r = d_r';





end

