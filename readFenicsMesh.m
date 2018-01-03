function [coordinates] = readFenicsMesh(file)
%This reads in a mesh from fenics xml

vertices = getChildNodes(item(getChildNodes(item(getChildNodes(...
    item(getChildNodes(xmlread(file)), 0)), 1)), 1));
nVertices = (vertices.getLength - 1)/2;


indices = zeros(1, nVertices);
coordinates = zeros(2, nVertices);

for vtx = 1:nVertices
    char_x = toCharArray(toString(item(getAttributes(...
        item(vertices, 2*vtx - 1)), 1)));
    coordinates(1, vtx) = str2double(strcat(char_x(4:24)'));
    char_y = toCharArray(toString(item(getAttributes(...
        item(vertices, 2*vtx - 1)), 2)));
    coordinates(2, vtx) = str2double(strcat(char_y(4:24)'));
end

end

