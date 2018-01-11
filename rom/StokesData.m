classdef StokesData
    %class for fine scale data of Stokes equation
    
    properties
        %Seldomly changed parameters are to bechanged here
        meshSize = 128
        nExclusions = [16, 257]   %[min, max] pos. number of circ. exclusions
        margins = [0, .025, 0, .025]    %[l., u.] margin for impermeable phase
        r_params = [.005, .015]    %[lo., up.] bound on random blob radius
        samples
        %base name of file path
        pathname = []
        
        %The properties below are saved as cell arrays (1 cell = 1 sample)
        X       %coordinates of mesh vertices as cell array
        P       %pressure at vertices
        U       %velocity at vertices
        cells   %cell-to-vertex map
        N_vertices_tot  %total number of vertices in data
        
        %Microstructural data, e.g. centers & radii of circular inclusions
        microstructData
        %Design matrix
        designMatrix
    end
    
    methods
        function [self] = StokesData(samples)
            %constructor
            self.samples = samples;
            for n = 1:numel(samples)
                self.designMatrix{n} = [];
            end
        end
        
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
        
        function [self] = readData(self, quantities)
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
            for n = self.samples
                file = matfile(char(strcat(self.pathname, 'solution',...
                    num2str(n), '.mat')));
                
                if any(quantities == 'x')
                    self.X{cellIndex} = file.x;
                end
                
                if any(quantities == 'p')
                    self.P{cellIndex} = file.p';
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
        
        function self = countVertices(self)
            cellIndex = 1;
            self.N_vertices_tot = 0;
            if isempty(self.P)
                self = self.readData('p');
            end
            for n = self.samples
                self.N_vertices_tot = self.N_vertices_tot +...
                    numel(self.P{cellIndex});
                cellIndex = cellIndex + 1;
            end
        end
        
        function [self] = evaluateFeatures(self, gridX, gridY)
            %Evaluates the feature functions
            if isempty(self.microstructData)
                self = self.readData('m');
            end
            
            %constant 1
            for n = 1:numel(self.samples)
                self.designMatrix{n} = [self.designMatrix{n},...
                    ones(numel(gridX)*numel(gridY), 1)];
            end
            
            %pore fraction
            for n = 1:numel(self.samples)
                phi = volumeFractionCircExclusions(...
                    self.microstructData{n}.diskCenters,...
                    self.microstructData{n}.diskRadii, gridX, gridY);
                self.designMatrix{n} = [self.designMatrix{n}, phi(:)];
            end
            
            %log pore fraction
            for n = 1:numel(self.samples)
                phi = log(volumeFractionCircExclusions(...
                    self.microstructData{n}.diskCenters,...
                    self.microstructData{n}.diskRadii, gridX, gridY));
                self.designMatrix{n} = [self.designMatrix{n}, phi(:)];
            end
            
            %Interface area
            for n = 1:numel(self.samples)
                phi = interfacePerVolume(...
                    self.microstructData{n}.diskCenters,...
                    self.microstructData{n}.diskRadii, gridX, gridY);
                self.designMatrix{n} = [self.designMatrix{n}, phi(:)];
            end
            
            %log interface area
            for n = 1:numel(self.samples)
                phi = log(interfacePerVolume(...
                    self.microstructData{n}.diskCenters,...
                    self.microstructData{n}.diskRadii, gridX, gridY) + eps);
                self.designMatrix{n} = [self.designMatrix{n}, phi(:)];
            end
        end
        
        function [triHandles, pltHandles, figHandle] = plotData(self, samples)
            %Plots the fine scale data and returns handles
            
            %Load data if not yet loaded
            if isempty(self.cells)
                self = self.readData('c');
            end
            if isempty(self.X)
                self = self.readData('x');
            end
            if isempty(self.P)
                self = self.readData('p');
            end
            if isempty(self.U)
                self = self.readData('u');
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

