classdef StokesData
    %class for fine scale data of Stokes equation
    
    properties
        %Seldomly changed parameters are to bechanged here
        meshSize = 128
        nExclusions = [256, 257]   %[min, max] pos. number of circ. exclusions
        margins = [0, .025, 0, .025]    %[l., u.] margin for impermeable phase
        r_params = [.005, .025]    %[lo., up.] bound on random blob radius
        %base name of file path
        pathname = []
        
        %The properties below are saved as cell arrays (1 cell = 1 sample)
        X       %coordinates of mesh vertices as cell array
        P       %pressure at vertices
        U       %velocity at vertices
        cells   %cell-to-vertex map
        
        %Microstructural data, e.g. centers & radii of circular inclusions
        microstructData
    end
    
    methods
        function [self] = setPathName(self)
            if isempty(self.pathname)
                self.pathname = strcat('/home/constantin/cluster/python/',...
                    'data/stokesEquation/meshes/');
                self.pathname = char(strcat(self.pathname, 'meshSize=',...
                    num2str(self.meshSize), '/nNonOverlapCircExcl=',...
                    num2str(self.nExclusions(1)), '-',...
                    num2str(self.nExclusions(2)),...
                    '/coordDist=uniform_margins=(',...
                    num2str(self.margins(1)), {', '},...
                    num2str(self.margins(2)), {', '},...
                    num2str(self.margins(3)), {', '},...
                    num2str(self.margins(4)), ')/radiiDist=uniform_',...
                    'r_params=(', num2str(self.r_params(1)),...
                    {', '}, num2str(self.r_params(2)), ')/'));
            end
        end
        
        function [self] = readData(self, samples, quantities)
            %Reads in Stokes equation data from fenics
            %samples:          samples to load
            %quantities:       identifier for the quantities to load,
            %                  'x' for vertex locations
            %                  'p' for pressure,
            %                  'u' for velocuty,
            %                  'c' for cell-to-vertex map
            %                  'm' for microstructural data
            
            self = self.setPathName;
            
            cellIndex = 1;
            for n = samples
                file = matfile(char(strcat(self.pathname, 'solution',...
                    num2str(n), '.mat')));
                
                if any(quantities == 'x')
                    self.X{cellIndex} = file.x;
                end
                
                if any(quantities == 'p')
                    self.P{cellIndex} = file.p;
                end
                
                if any(quantities == 'u')
                    self.U{cellIndex} = file.u;
                end
                
                if any(quantities == 'c')
                    cellfile = matfile(char(strcat(self.pathname, 'mesh',...
                    num2str(n), '.mat')));
                    self.cells{cellIndex} = cellfile.cells;
                end
                
                if any(quantities == 'm')
                    datafile = char(strcat(self.pathname,...
                        'microstructureInformation', num2str(n), '.mat'));
                    self.microstructData{cellIndex} = load(datafile);
                end
                cellIndex = cellIndex + 1;
            end
        end
        
        function [self] = evaluateFeatures(self)
            %Evaluates the feature functions
        end
        
        function [triHandles, pltHandles, figHandle] = plotData(self, samples)
            %Plots the fine scale data and returns handles
            
            %Load data if not yet loaded
            if isempty(self.cells)
                self = self.readData(samples, 'c');
            end
            if isempty(self.X)
                self = self.readData(samples, 'x');
            end
            if isempty(self.P)
                self = self.readData(samples, 'p');
            end
            if isempty(self.U)
                self = self.readData(samples, 'u');
            end
            
            
            figHandle = figure;
            pltIndex = 1;
            N = numel(samples);
            for n = samples
                %pressure field
                pltHandles(1, pltIndex) = subplot(2, N, pltIndex);
                triHandles(1, pltIndex) =...
                    trisurf(self.cells{pltIndex}, self.X{pltIndex}(:, 1),...
                    self.X{pltIndex}(:, 2), self.P{pltIndex});
                triHandles(1, pltIndex).LineStyle = 'none';
                axis square;
                view(2);
                grid off;
                box on;
                xticks({});
                yticks({});
                cbp = colorbar;
                
                %velocity field (norm)
                u_norm = sqrt(sum(self.U{pltIndex}.^2));
                pltHandles(2, pltIndex) = subplot(2, N, pltIndex + N);
                triHandles(2, pltIndex) = trisurf(self.cells{pltIndex},...
                   self.X{pltIndex}(:, 1), self.X{pltIndex}(:, 2), u_norm);
                triHandles(2, pltIndex).LineStyle = 'none';
                axis square;
                view(2);
                grid off;
                box on;
                xticks({});
                yticks({});
                cbu = colorbar;
                
                pltIndex = pltIndex + 1;
            end
        end
    end
end

