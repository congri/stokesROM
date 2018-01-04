function [value, cell_index, cell_dof_index] =...
    readFenicsSolution(file)
%This reads in Stokes solution from fenics xml

xmlfile = xmlread(file);

function_data = item(item(xmlfile, 0), 1);
nDof = (getLength(function_data) - 1)/2;


cell_dof_index = zeros(nDof, 1);
cell_index = zeros(nDof, 1);
value = zeros(nDof, 1);
for dof = 1:nDof
    dofAttr = getAttributes(item(function_data, 2*dof - 1));
    
    cell_dof_index_char = toCharArray(toString(item(dofAttr, 0)))';
    cell_dof_index(dof) = str2double(cell_dof_index_char(17:(end - 1)));
    
    cell_index_char = toCharArray(toString(item(dofAttr, 1)))';
    cell_index(dof) = str2double(cell_index_char(13:(end - 1)));
        
    value_char = toCharArray(toString(item(dofAttr, 3)))';
    value(dof) = str2double(value_char(8:(end - 1)));
end

%Conversion to matlab indexing
cell_dof_index = cell_dof_index + 1;
cell_index = cell_index + 1;


end