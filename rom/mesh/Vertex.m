classdef Vertex < handle
    %Mesh vertex class
    
    properties(SetAccess = private)
        coordinates                 %coordinates of vertex
        cells                       %cells attached to vertex
        edges                       %edges attached to vertex
    end
    
    methods
        function self = Vertex(coordinates)
            %Constructor
            self.coordinates = coordinates;
        end
        
        function self = add_cell(self, cell)
            self.cells{end + 1} = cell;
        end
        
        function self = add_edge(self, edge)
            self.edges{end + 1} = edge;
        end
        
        function [p] = plot(self, n, fig)
            if nargin > 2
                figure(fig);
            end
            
            p = plot(self.coordinates(1), self.coordinates(2), 'bx',...
                'linewidth', 2, 'markersize', 8);
            text(self.coordinates(1) + .02, self.coordinates(2) + .02, num2str(n), 'color', 'b');
        end
    end
end

