classdef StokesData
    %class for fine scale data of Stokes equation
    
    properties
        %Seldomly changed parameters are to bechanged here
        meshSize = 128
        nExclusions = [2048, 2049]   %[min, max] pos. number of circ. exclusions
        margins = [0, .03, 0, .03]    %[l., u.] margin for impermeable phase
        r_params = [.001, .01]    %[lo., up.] bound on random blob radius
        samples
        %base name of file path
        pathname = []
        
        %The properties below are saved as cell arrays (1 cell = 1 sample)
        X       %coordinates of mesh vertices as cell array
        P       %pressure at vertices
        U       %velocity at vertices
        cells   %cell-to-vertex map
        cellOfVertex   %Mapping from vertex to sq. (macro-)cell
        N_vertices_tot  %total number of vertices in data
        
        %Microstructural data, e.g. centers & radii of circular inclusions
        microstructData
        %Flow boundary conditions; C++ string
        u_bc
        %Design matrix
        designMatrix
    end
    
    methods
        function [self] = StokesData(samples, u_bc)
            %constructor
            self.samples = samples;
            self.u_bc = u_bc;
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
                foldername = char(strcat(self.pathname, self.u_bc{1}, '_',...
                    self.u_bc{2}));
                filename = char(strcat(foldername, '/solution',...
                    num2str(n), '.mat'));
                file = matfile(filename);
                
                if exist(filename, 'file')
                    
                    if any(quantities == 'x')
                        self.X{cellIndex} = file.x;
                    end
                    
                    if any(quantities == 'p')
                        rescale_p = true;
                        if rescale_p
                            p_temp = file.p';
                            p_origin = p_temp(all((file.x == [0, 0])'));
                            p_temp = p_temp - p_origin;
                        else
                            p_temp = file.p';
                        end
                        self.P{cellIndex} = p_temp;
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
                else
                    self.samples(self.samples == n) = [];
                    warning(strcat(filename, 'not found. Skipping sample.'))
                end
            end
        end
        
        function self = countVertices(self)
            self.N_vertices_tot = 0;
            if isempty(self.P)
                self = self.readData('p');
            end
            for cellIndex = 1:numel(self.P)
                self.N_vertices_tot = self.N_vertices_tot +...
                    numel(self.P{cellIndex});
            end
        end
        
        function self = vtxToCell(self, gridX, gridY)
            %Mapping from vertex to macro-cell given by grid vectors gridX,
            %gridY
            cumsumX = cumsum(gridX);
            cumsumY = cumsum(gridY);
            
            Nx = numel(gridX);
            Ny = numel(gridY);
            
            if isempty(self.X)
                self = self.readData('x');
            end
            
            for n = 1:numel(self.X)
                self.cellOfVertex{n} = zeros(size(self.X{n}, 1), 1);
                for vtx = 1:size(self.X{n}, 1)
                    nx = 1;
                    while(self.X{n}(vtx, 1) > cumsumX(nx))
                        nx = nx + 1;
                    end
                    ny = 1;
                    while(self.X{n}(vtx, 2) > cumsumY(ny))
                        ny = ny + 1;
                    end
                    self.cellOfVertex{n}(vtx) = nx + (ny - 1)*Nx;
                end
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
                figure(figHandle);
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
                cbp.Label.String = 'pressure p';
                
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
                cbu.Label.String = 'velocity u';
                
                pltIndex = pltIndex + 1;
            end
        end
    end
end

