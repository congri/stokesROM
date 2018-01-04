%test file to read in xml data

xmlfile = xmlread('/home/constantin/python/data/stokesEquation/meshes/meshSize=128/nCircExcl=256/coordDist=uniform/radiiDist=uniform_r_params=(0.005, 0.03)/fullSolution0.xml');


dolfin = item(xmlfile, 0)
function_data = item(dolfin, 1)
getLength(function_data)
dof1 = getAttributes(item(function_data, 1))
N = getLength(dof1)

cell_dof_index_char = toCharArray(toString(item(dof1, 0)))';
cell_dof_index = str2double(cell_dof_index_char(17:(end - 1)))

cell_index_char = toCharArray(toString(item(dof1, 1)))';
cell_index = str2double(cell_index_char(13:(end - 1)))

index_char = toCharArray(toString(item(dof1, 2)))';
index = str2double(index_char(8:(end - 1)))

value_char = toCharArray(toString(item(dof1, 3)))';
value = str2double(value_char(8:(end - 1)))
