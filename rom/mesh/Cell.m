classdef Cell < handle
    %Mesh cell class
    
    properties(SetAccess = private)
        vertices                %adjacent vertices. Have to be sorted according
                                %to local vertex number!(counterclockwise
                                %starting from lower left corner)
        edges                   %adjacent edges. Have to be sorted according to
                                %local edge number! (counterclockwise starting 
                                %from lower left corner)
        surface                 %surface of cell element
        centroid                %centroid of cell
    end
    
    methods
        function self = Cell(vertices, edges)
            %Constructor
            %Vertices and edges must be sorted according to local
            %vertex/edge number!
            self.vertices = vertices;
            self.edges = edges;
            
            %Compute centroid
            self.centroid = 0;
            for n = 1:numel(self.vertices)
                self.centroid = self.centroid + self.vertices{n}.coordinates;
            end
            self.centroid = self.centroid/numel(self.vertices);
        end
        
        function delete_edges(self, indices)
            %Deletes edges according to indices
            for i = indices
                delete(self.edges{i});
                self.edges{i} = [];
            end
        end
    end
end

