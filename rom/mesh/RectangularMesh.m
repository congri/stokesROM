classdef RectangularMesh < Msh
    %Class for mesh consisting of generic rectangles

    
    methods
        function self = RectangularMesh(self)
            %constructor
        end
        
        function self = split_cell(self, cll)
            %Splits the rectangular cell cll
            
            %Create new vertices
            new_vertices{1} = self.create_vertex(cll.centroid);
            lo_coord = .5*(cll.edges{1}.vertices{1}.coordinates + ...
                cll.edges{1}.vertices{2}.coordinates);
            new_vertices{2} = self.create_vertex(lo_coord);
            r_coord = .5*(cll.edges{2}.vertices{1}.coordinates + ...
                cll.edges{2}.vertices{2}.coordinates);
            new_vertices{3} = self.create_vertex(r_coord);
            up_coord = .5*(cll.edges{3}.vertices{1}.coordinates + ...
                cll.edges{3}.vertices{2}.coordinates);
            new_vertices{4} = self.create_vertex(up_coord);
            le_coord = .5*(cll.edges{4}.vertices{1}.coordinates + ...
                cll.edges{4}.vertices{2}.coordinates);
            new_vertices{5} = self.create_vertex(le_coord);
            
            %Create new edges. Go around old cell, then middle cross
            %go around
            new_edges{1} = self.create_edge(cll.vertices{1}, new_vertices{2});
            new_edges{2} = self.create_edge(new_vertices{2}, cll.vertices{2});
            new_edges{3} = self.create_edge(cll.vertices{2}, new_vertices{3});
            new_edges{4} = self.create_edge(new_vertices{3}, cll.vertices{3});
            new_edges{5} = self.create_edge(cll.vertices{3}, new_vertices{4});
            new_edges{6} = self.create_edge(new_vertices{4}, cll.vertices{4});
            new_edges{7} = self.create_edge(cll.vertices{4}, new_vertices{5});
            new_edges{8} = self.create_edge(new_vertices{5}, cll.vertices{1});
            %middle cross
            new_edges{9} = self.create_edge(new_vertices{1}, new_vertices{2});
            new_edges{10} = self.create_edge(new_vertices{1}, new_vertices{3});
            new_edges{11} = self.create_edge(new_vertices{1}, new_vertices{4});
            new_edges{12} = self.create_edge(new_vertices{1}, new_vertices{5});
            
            
            %Lower left subcell
            vertices = {cll.vertices{1}, new_vertices{2}, new_vertices{1},...
                new_vertices{5}};
            edges = {new_edges{1}, new_edges{9}, new_edges{12}, new_edges{8}};
            self.create_cell(vertices, edges);
            
            %Lower right subcell
            vertices = {new_vertices{2}, cll.vertices{2}, new_vertices{3},...
                new_vertices{1}};
            edges = {new_edges{2}, new_edges{3}, new_edges{10}, new_edges{9}};
            self.create_cell(vertices, edges);
            
            %Upper right subcell
            vertices = {new_vertices{1}, new_vertices{3}, cll.vertices{3},...
                new_vertices{4}};
            edges = {new_edges{10}, new_edges{4}, new_edges{5}, new_edges{11}};
            self.create_cell(vertices, edges);
            
            %Upper left subcell
            vertices = {new_vertices{5}, new_vertices{1}, new_vertices{4},...
                cll.vertices{4}};
            edges = {new_edges{12}, new_edges{11}, new_edges{6}, new_edges{7}};
            self.create_cell(vertices, edges);
            
            %Delete old edges and cell
            cll.delete_edges(1:4);
            delete(cll);
            
            %Update statistics
            self.nEdges = self.nEdges - 4;
            self.nCells = self.nCells - 1;
        end
        
    end
end

