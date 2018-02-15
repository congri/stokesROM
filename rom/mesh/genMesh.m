%Test script to generate object oriented mesh object

clear;

%grid vectors for x and y
gridX = (1/1)*ones(1, 1);
gridY = gridX;

vtxX = cumsum([0, gridX]);
vtxY = cumsum([0, gridY]);

msh = RectangularMesh;

%Set vertices
i = 1;
for y = vtxY
    for x = vtxX
        msh.create_vertex([x, y]);
        i = i + 1;
    end
end

%Set edges by hand
msh.create_edge(msh.vertices{1}, msh.vertices{2});
msh.create_edge(msh.vertices{2}, msh.vertices{4});
msh.create_edge(msh.vertices{4}, msh.vertices{3});
msh.create_edge(msh.vertices{3}, msh.vertices{1});

%Set cell by hand
vtx = {msh.vertices{1} msh.vertices{2} msh.vertices{4} msh.vertices{3}};
msh.create_cell(vtx, msh.edges);



%plot mesh
msh.plotMesh;

msh.split_cell(msh.cells{1});

%plot mesh
msh.plotMesh;

msh.split_cell(msh.cells{3});
msh.plotMesh;

msh.split_cell(msh.cells{9});
msh.plotMesh;

msh.split_cell(msh.cells{10});
msh.plotMesh;


    

