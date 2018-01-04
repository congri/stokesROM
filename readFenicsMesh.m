function [vtxCoordinates, cllIndices] = readFenicsMesh(file)
%This reads in a mesh from fenics xml
%   vtxCoordinates:     array of vertex coordinates
%   cllIndices:         array of vertex indices making a cell

%% Vertices
xmlfile = xmlread(file);
vertices = item(item(item(xmlfile, 0), 1), 1);
nVertices = (vertices.getLength - 1)/2;

vtxCoordinates = zeros(2, nVertices);
for vtx = 1:nVertices
    char_x = toCharArray(toString(item(getAttributes(...
        item(vertices, 2*vtx - 1)), 1)));
    vtxCoordinates(1, vtx) = str2double(strcat(char_x(4:24)'));
    char_y = toCharArray(toString(item(getAttributes(...
        item(vertices, 2*vtx - 1)), 2)));
    vtxCoordinates(2, vtx) = str2double(strcat(char_y(4:24)'));
end


%% Cells
cells = item(item(item(xmlfile, 0), 1), 3);
nCells = (cells.getLength - 1)/2;

cllIndices = zeros(nCells, 3);
for cll = 1:nCells
    char_v0 = toCharArray(toString(item(getAttributes(...
        item(cells, 2*cll - 1)), 1)));
    cllIndices(cll, 1) = str2double(strcat(char_v0(5:(end - 1))'));
    char_v1 = toCharArray(toString(item(getAttributes(...
        item(cells, 2*cll - 1)), 2)));
    cllIndices(cll, 2) = str2double(strcat(char_v1(5:(end - 1))'));
    char_v2 = toCharArray(toString(item(getAttributes(...
        item(cells, 2*cll - 1)), 3)));
    cllIndices(cll, 3) = str2double(strcat(char_v2(5:(end - 1))'));
end
cllIndices = cllIndices + 1;    %conversion to matlab indices


debug = false;
if debug
    %plot loaded mesh
    figure
    tp = triplot(cllIndices, vtxCoordinates(1, :), vtxCoordinates(2, :));
    axis square
    tp.Color = 'k';
    grid off
    xticks({})
    yticks({})
end


end

