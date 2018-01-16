function [X, P, U, C] = readData(N)
%Reads in Stokes equation data from fenics

%   N:          samples to load

%Seldomly changed parameters are to bechanged here
meshSize = 128;
nExclusions = [256, 2049];   %[min, max] possible number of circ. exclusions
c_params = [.025, .975];    %[lower, upper] margin for impermeable phase
r_params = [.005, .025];    %[lower, upper] bound on uni. dist. on blob radius
pathname = '/home/constantin/cluster/python/data/stokesEquation/meshes/';

pathname = strcat(pathname, 'meshSize=', num2str(meshSize), ...
    '/nNonOverlapCircExcl=', num2str(nExclusions(1)), '-', ...
    num2str(nExclusions(2)), '/coordDist=uniform_c_params=(', ...
    num2str(c_params(1)), {', '}, num2str(c_params(2)), ...
    ')/radiiDist=uniform_','r_params=(', num2str(r_params(1)), {', '}, ...
    num2str(r_params(2)), ')/');

for n = N
   load(char(strcat(pathname, 'solution', num2str(n), '.mat')));
   X{n + 1} = x;
   P{n + 1} = p;
   if nargout > 2
       U{n + 1} = u;
   end
   if nargout > 3
       load(char(strcat(pathname, 'mesh', num2str(n), '.mat')), 'cells');
       C{n + 1} = cells;
   end
end

end

