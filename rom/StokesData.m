classdef StokesData < handle
    %class for fine scale data of Stokes equation
    
    properties
        %Seldomly changed parameters are to bechanged here
        meshSize = 128
        numberParams = [5.0, 1.0]   %[min, max] pos. number of circ. exclusions
        numberDist = 'logn';
        margins = [-1, .01, -1, .01]    %[l., u.] margin for impermeable phase
        r_params = [-4.0, .7]    %[lo., up.] bound on random blob radius
        coordDist = 'gauss'
        coordDist_mu = '[0.4, 0.6]'   %only for gauss
        coordDist_cov = '[[0.035, 0.0], [0.0, 0.08]]'
        radiiDist = 'logn'
        samples
        %base name of file path
        pathname = []
        
        %The properties below are saved as cell arrays (1 cell = 1 sample)
        X       %coordinates of mesh vertices as cell array
        X_interp%coordinates of interpolation mesh, if data is interpolated
        input_bitmap  %bitmap of microstr.; true is pore, false is solid phase
                %onto regular mesh
        P       %pressure at vertices
        U       %velocity at vertices
        cells   %cell-to-vertex map
        cellOfVertex   %Mapping from vertex to cell of S (p_cf variance)
        N_vertices_tot  %total number of vertices in data
        
        %Microstructural data, e.g. centers & radii of circular inclusions
        microstructData
        %Flow boundary conditions; C++ string
        u_bc
        %Design matrix
        designMatrix
    end
    
    methods
        function self = StokesData(samples, u_bc)
            %constructor
            self.samples = samples;
            self.u_bc = u_bc;
            for n = 1:numel(samples)
                self.designMatrix{n} = [];
            end
        end
        
        function setPathName(self)
            if isempty(self.pathname)
                self.pathname = strcat('/home/constantin/python/',...
                    'data/stokesEquation/meshes/');
                
%                 self.pathname = char(strcat(self.pathname, 'meshSize=',...
%                     num2str(self.meshSize), '/nNonOverlapCircExcl=',...
%                     num2str(self.nExclusions(1)), '-',...
%                     num2str(self.nExclusions(2)),...
%                     '/coordDist=', self.coordDist, '_margins=(',...
%                     num2str(self.margins(1)), {', '},...
%                     num2str(self.margins(2)), {', '},...
%                     num2str(self.margins(3)), {', '},...
%                     num2str(self.margins(4)), ')/radiiDist=', self.radiiDist,...
%                     '_r_params=(', num2str(self.r_params(1)),...
%                     {', '}, num2str(self.r_params(2)), ')/'));
                self.pathname = char(strcat(self.pathname, 'meshSize=',...
                    num2str(self.meshSize), '/nNonOverlapCircExcl=',...
                    self.numberDist,...
                    sprintf('%.1f', self.numberParams(1)), '-', ...
                    sprintf('%.1f', self.numberParams(2)), ...
                    '/coordDist=', self.coordDist, '_mu=', self.coordDist_mu,...
                    'cov=', self.coordDist_cov, '_margins=(',...
                    num2str(self.margins(1)), {', '},...
                    num2str(self.margins(2)), {', '},...
                    num2str(self.margins(3)), {', '},...
                    num2str(self.margins(4)), ')/radiiDist=', self.radiiDist,...
                    '_r_params=(', sprintf('%.1f', self.r_params(1)),...
                    {', '}, num2str(self.r_params(2)), ')/'));
            end
        end
        
        function readData(self, quantities)
            %Reads in Stokes equation data from fenics
            %samples:          samples to load
            %quantities:       identifier for the quantities to load,
            %                  'x' for vertex locations
            %                  'p' for pressure,
            %                  'u' for velocuty,
            %                  'c' for cell-to-vertex map
            %                  'm' for microstructural data
            
            self.setPathName;
            
            cellIndex = 1;
            for n = self.samples
                foldername = char(strcat(self.pathname, self.u_bc{1}, '_',...
                    self.u_bc{2}));
                filename = char(strcat(foldername, '/solution',...
                    num2str(n), '.mat'));
                file = matfile(filename);
                
                if exist(filename, 'file')
                    
                    if contains(quantities, 'x')
                        self.X{cellIndex} = file.x;
                    end
                    
                    if contains(quantities, 'p')
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
                    
                    if contains(quantities, 'u')
                        self.U{cellIndex} = file.u;
                    end
                    
                    if contains(quantities, 'c')
                        cellfile = matfile(char(strcat(self.pathname, 'mesh',...
                            num2str(n), '.mat')));
                        self.cells{cellIndex} = cellfile.cells;
                    end
                    
                    if contains(quantities, 'm')
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
        
        function interpolate(self, fineGridX, fineGridY, interpolationMode, ...
                smoothingParameter, boundarySmoothingPixels)
            %Interpolates finescale data onto a regular rectangular grid
            %specified by fineGridX, fineGridY
            
            if nargin < 3
                fineGridY = fineGridX;
            end
            if nargin < 4
                interpolationMode = 'linear';
            end
            if nargin < 5
                smoothingParameter = [];
            end
            
            fineGridX = [0, cumsum(fineGridX)];
            fineGridY = [0, cumsum(fineGridY)];
            
            if isempty(self.X)
                self.readData('x');
            end
            
            %Specify query grid
            [xq, yq] = meshgrid(fineGridX, fineGridY);
            for n = 1:numel(self.P)
                if ~isempty(self.P)
                    p_interp = griddata(self.X{n}(:, 1), self.X{n}(:, 2), ...
                        self.P{n}, xq(:), yq(:), interpolationMode);
                    %replace original data by interpolated data
                    if ~isempty(smoothingParameter)
                        p_interp = reshape(p_interp, numel(fineGridX), ...
                            numel(fineGridY));
                        if boundarySmoothingPixels > 0
                            %only smooth boundary
                            p_temp = imgaussfilt(p_interp, smoothingParameter);
                            p_interp(1:boundarySmoothingPixels, :) = ...
                                p_temp(1:boundarySmoothingPixels, :);
                            p_interp((end - boundarySmoothingPixels):end,:)=...
                                p_temp((end - boundarySmoothingPixels):end,:);
                            p_interp(:, 1:boundarySmoothingPixels) = ...
                                p_temp(:, 1:boundarySmoothingPixels);
                            p_interp(:, (end - boundarySmoothingPixels):end)=...
                                p_temp(:, (end - boundarySmoothingPixels):end);
                        else
                            p_interp= imgaussfilt(p_interp, smoothingParameter);
                        end
                        p_interp = p_interp(:);
                    end
                    self.P{n} = p_interp;
                end
                
                if ~isempty(self.U)
                    u_interp_x = griddata(self.X{n}(:, 1), self.X{n}(:, 2), ...
                        self.U{n}(1, :), xq(:), yq(:), interpolationMode);
                    u_interp_y = griddata(self.X{n}(:, 1), self.X{n}(:, 2), ...
                        self.U{n}(2, :), xq(:), yq(:), interpolationMode);
                    %replace original data by interpolated data
                    self.U{n} = [];
                    self.U{n} = [u_interp_x, u_interp_y];
                end
            end
            self.X_interp{1} = [xq(:), yq(:)];
        end
        
        function dataVar = computeDataVariance(self, samples, quantity,...
                fineGridX, fineGridY, interpolationMode, smoothingParameter)
            %Computes variance of Stokes data over whole data set
            %keep in mind python indexing for samples
            
            self.samples = samples;
            disp('Reading in data...')
            self.readData(quantity);
            disp('... data read in.')
            
            if isempty(self.X_interp)
                %Data has not yet been interpolated onto a regular grid
                if nargin < 7
                    %no smoothing
                    smoothingParameter = [];
                end
                if nargin < 6
                    interpolationMode = 'linear';
                end
                if nargin < 5
                    fineGridY = fineGridX;
                end
                self.interpolate(fineGridX, fineGridY, interpolationMode,...
                    smoothingParameter);
            end
            
            %Compute first and second moments
            meanQuantity = 0;
            meanSquaredQuantity = 0;
            for n = 1:numel(samples)
                if strcmp(quantity, 'p')
                    meanQuantity = (1/n)*((n - 1)*meanQuantity + self.P{n});
                    meanSquaredQuantity =...
                        (1/n)*((n - 1)*meanSquaredQuantity + self.P{n}.^2);
                elseif strcmp(quantity, 'u')
                    meanQuantity = (1/n)*((n - 1)*meanQuantity + self.U{n}(:));
                    meanSquaredQuantity =...
                        (1/n)*((n - 1)*meanSquaredQuantity + self.U{n}(:).^2);
                else
                    error('unknown quantity');
                end
            end
            dataVar = meanSquaredQuantity - meanQuantity.^2;
            meanDataVar = mean(dataVar);
        end
        
        function input2bitmap(self, gridX, gridY)
            %Converts input microstructures to bitmap images
            %Feed in grid vectors for vertex coordinates, not elements!
            %first index in input_bitmap is x-index!
            
            if isempty(self.X)
                self.readData('x');
            end
            if isempty(self.microstructData)
                self.readData('m');
            end
            
            if nargin < 3
                gridY = gridX;
            end
            
            %gridX, gridY must be row vectors
            if size(gridX, 1) > 1
                gridX = gridX';
            end
            if size(gridX, 1) > 1
                gridY = gridY';
            end
            
            
            centroids_x = movmean(gridX, 2); centroids_x = centroids_x(2:end);
            centroids_y = movmean(gridY, 2); centroids_y = centroids_y(2:end);
            
            for n = 1:numel(self.X)
                self.input_bitmap{n} = true(numel(gridX) - 1, numel(gridY) - 1);
                %Loop over all element centroids and check if they are within 
                %the domain or not
                    dist_x_sq = (self.microstructData{n}.diskCenters(:, 1)...
                        - centroids_x).^2;
                    dist_y_sq = (self.microstructData{n}.diskCenters(:, 2)...
                        - centroids_y).^2;
                    for circle = 1:numel(self.microstructData{n}.diskRadii)
                        %set pixels inside circle to false
                        dist_sq = dist_x_sq(circle, :)' + dist_y_sq(circle, :);
                        self.input_bitmap{n}(dist_sq < self...
                            .microstructData{n}.diskRadii(circle)^2) = false;
                    end
            end
        end
        
        function countVertices(self)
            self.N_vertices_tot = 0;
            if isempty(self.P)
                self = self.readData('p');
            end
            for cellIndex = 1:numel(self.P)
                self.N_vertices_tot = self.N_vertices_tot +...
                    numel(self.P{cellIndex});
            end
        end
        
        function vtxToCell(self, gridX, gridY, interpolationMode)
            %Mapping from vertex to cell of rectangular grid specified by
            %grid vectors gridX, gridY
            if nargin < 4
                interpolationMode = false;
            end
            if nargin < 3
                gridY = gridX;
            end
            cumsumX = cumsum(gridX);
            cumsumX(end) = cumsumX(end) + 1e-12;  %include vertices on boundary
            cumsumY = cumsum(gridY);
            cumsumY(end) = cumsumY(end) + 1e-12;  %include vertices on boundary
            
            Nx = numel(gridX);
            
            if isempty(self.X)
                self = self.readData('x');
            end
            if any(interpolationMode)
                if isempty(self.X_interp)
                    self = self.interpolate(gridX, gridY, interpolationMode);
                end
                X = self.X_interp;
            else
                X = self.X;
            end
            
            for n = 1:numel(X)
                self.cellOfVertex{n} = zeros(size(X{n}, 1), 1);
                for vtx = 1:size(X{n}, 1)
                    nx = 1;
                    while(X{n}(vtx, 1) > cumsumX(nx))
                        nx = nx + 1;
                    end
                    ny = 1;
                    while(X{n}(vtx, 2) > cumsumY(ny))
                        ny = ny + 1;
                    end
                    self.cellOfVertex{n}(vtx) = nx + (ny - 1)*Nx;
                end
            end
        end
        
        function evaluateFeatures(self, gridRF)
            %Evaluates the feature functions
            if isempty(self.microstructData)
                self.readData('m');
            end
            
            %constant 1
            for n = 1:numel(self.samples)
                self.designMatrix{n} = [self.designMatrix{n},...
                    ones(gridRF.nCells, 1)];
            end
            
            %pore fraction
            for n = 1:numel(self.samples)
                phi = volumeFractionCircExclusions(...
                    self.microstructData{n}.diskCenters,...
                    self.microstructData{n}.diskRadii, gridRF);
                self.designMatrix{n} = [self.designMatrix{n}, phi(:)];
            end
            
            %log pore fraction
            for n = 1:numel(self.samples)
                phi = log(volumeFractionCircExclusions(...
                    self.microstructData{n}.diskCenters,...
                    self.microstructData{n}.diskRadii, gridRF) + eps);
                self.designMatrix{n} = [self.designMatrix{n}, phi(:)];
            end
            
            %Interface area
            for n = 1:numel(self.samples)
                phi = interfacePerVolume(...
                    self.microstructData{n}.diskCenters,...
                    self.microstructData{n}.diskRadii, gridRF);
                self.designMatrix{n} = [self.designMatrix{n}, phi(:)];
            end
            
            %log interface area
            for n = 1:numel(self.samples)
                phi = log(interfacePerVolume(...
                    self.microstructData{n}.diskCenters,...
                    self.microstructData{n}.diskRadii, gridRF) + eps);
                self.designMatrix{n} = [self.designMatrix{n}, phi(:)];
            end
            
            %exp interface area
            for n = 1:numel(self.samples)
                phi = exp(interfacePerVolume(...
                    self.microstructData{n}.diskCenters,...
                    self.microstructData{n}.diskRadii, gridRF));
                self.designMatrix{n} = [self.designMatrix{n}, phi(:)];
            end
            
            %sqrt interface area
            for n = 1:numel(self.samples)
                phi = sqrt(interfacePerVolume(...
                    self.microstructData{n}.diskCenters,...
                    self.microstructData{n}.diskRadii, gridRF) + eps);
                self.designMatrix{n} = [self.designMatrix{n}, phi(:)];
            end
            
            %square interface area
            for n = 1:numel(self.samples)
                phi = (interfacePerVolume(...
                    self.microstructData{n}.diskCenters,...
                    self.microstructData{n}.diskRadii, gridRF)).^2;
                self.designMatrix{n} = [self.designMatrix{n}, phi(:)];
            end
            
            %cube interface area
            for n = 1:numel(self.samples)
                phi = (interfacePerVolume(...
                    self.microstructData{n}.diskCenters,...
                    self.microstructData{n}.diskRadii, gridRF)).^3;
                self.designMatrix{n} = [self.designMatrix{n}, phi(:)];
            end
            
            %mean distance between disk edges
            for n = 1:numel(self.samples)
                phi = diskDistance(self.microstructData{n}.diskCenters,...
                    self.microstructData{n}.diskRadii, gridRF, 'mean',...
                    'edge2edge');
                self.designMatrix{n} = [self.designMatrix{n}, phi(:)];
            end
            
            %mean distance between disks
            for n = 1:numel(self.samples)
                phi = diskDistance(self.microstructData{n}.diskCenters,...
                    self.microstructData{n}.diskRadii, gridRF, 'mean', 2);
                self.designMatrix{n} = [self.designMatrix{n}, phi(:)];
            end
            
            %min distance between disks
            for n = 1:numel(self.samples)
                phi = diskDistance(self.microstructData{n}.diskCenters,...
                    self.microstructData{n}.diskRadii, gridRF, 'min', 2);
                self.designMatrix{n} = [self.designMatrix{n}, phi(:)];
            end
            
            %specific surface is the same as interfacePerVolume
%             %specific surface of non-overlap. polydis. spheres
%             for n = 1:numel(self.samples)
%                 phi = specificSurface(self.microstructData{n}.diskCenters,...
%                     self.microstructData{n}.diskRadii, gridRF);
%                 self.designMatrix{n} = [self.designMatrix{n}, phi(:)];
%             end
%             
%             %log specific surface of non-overlap. polydis. spheres
%             for n = 1:numel(self.samples)
%                 phi =log(specificSurface(self.microstructData{n}.diskCenters,...
%                     self.microstructData{n}.diskRadii, gridRF) + eps);
%                 self.designMatrix{n} = [self.designMatrix{n}, phi(:)];
%             end
            
            %pore lin. path for non-overlap. polydis. spheres
            %last entry is distance
            for n = 1:numel(self.samples)
                phi = matrixLinealPath(self.microstructData{n}.diskCenters,...
                    self.microstructData{n}.diskRadii, gridRF, .1);
                self.designMatrix{n} = [self.designMatrix{n}, phi(:)];
            end
            
            for n = 1:numel(self.samples)
                phi = matrixLinealPath(self.microstructData{n}.diskCenters,...
                    self.microstructData{n}.diskRadii, gridRF, .08);
                self.designMatrix{n} = [self.designMatrix{n}, phi(:)];
            end
            
            for n = 1:numel(self.samples)
                phi = matrixLinealPath(self.microstructData{n}.diskCenters,...
                    self.microstructData{n}.diskRadii, gridRF, .06);
                self.designMatrix{n} = [self.designMatrix{n}, phi(:)];
            end
            
            for n = 1:numel(self.samples)
                phi = matrixLinealPath(self.microstructData{n}.diskCenters,...
                    self.microstructData{n}.diskRadii, gridRF, .04);
                self.designMatrix{n} = [self.designMatrix{n}, phi(:)];
            end
            
            for n = 1:numel(self.samples)
                phi = matrixLinealPath(self.microstructData{n}.diskCenters,...
                    self.microstructData{n}.diskRadii, gridRF, .02);
                self.designMatrix{n} = [self.designMatrix{n}, phi(:)];
            end
            
            %Chord length densities
            dist = .1;
            for n = 1:numel(self.samples)
                phi = chordLengthDensity(self.microstructData{n}.diskCenters,...
                    self.microstructData{n}.diskRadii, gridRF, dist);
                self.designMatrix{n} = [self.designMatrix{n}, phi(:)];
            end
            dist = .08;
            for n = 1:numel(self.samples)
                phi = chordLengthDensity(self.microstructData{n}.diskCenters,...
                    self.microstructData{n}.diskRadii, gridRF, dist);
                self.designMatrix{n} = [self.designMatrix{n}, phi(:)];
            end
            dist = .06;
            for n = 1:numel(self.samples)
                phi = chordLengthDensity(self.microstructData{n}.diskCenters,...
                    self.microstructData{n}.diskRadii, gridRF, dist);
                self.designMatrix{n} = [self.designMatrix{n}, phi(:)];
            end
            dist = .02;
            for n = 1:numel(self.samples)
                phi = chordLengthDensity(self.microstructData{n}.diskCenters,...
                    self.microstructData{n}.diskRadii, gridRF, dist);
                self.designMatrix{n} = [self.designMatrix{n}, phi(:)];
            end
            
            %Nearest surface functions
            dist = .05;
            for n = 1:numel(self.samples)
                [e_v, h_v] = voidNearestSurfaceExclusion(...
                    self.microstructData{n}.diskCenters,...
                    self.microstructData{n}.diskRadii,...
                    gridRF, dist);
                self.designMatrix{n} = [self.designMatrix{n}, e_v(:), h_v];
            end
            dist = .04;
            for n = 1:numel(self.samples)
                [e_v, h_v] = voidNearestSurfaceExclusion(...
                    self.microstructData{n}.diskCenters,...
                    self.microstructData{n}.diskRadii,...
                    gridRF, dist);
                self.designMatrix{n} = [self.designMatrix{n}, e_v(:), h_v];
            end
            dist = .03;
            for n = 1:numel(self.samples)
                [e_v, h_v] = voidNearestSurfaceExclusion(...
                    self.microstructData{n}.diskCenters,...
                    self.microstructData{n}.diskRadii,...
                    gridRF, dist);
                self.designMatrix{n} = [self.designMatrix{n}, e_v(:), h_v];
            end
            dist = .02;
            for n = 1:numel(self.samples)
                [e_v, h_v] = voidNearestSurfaceExclusion(...
                    self.microstructData{n}.diskCenters,...
                    self.microstructData{n}.diskRadii,...
                    gridRF, dist);
                self.designMatrix{n} = [self.designMatrix{n}, e_v(:), h_v];
            end
            dist = .01;
            for n = 1:numel(self.samples)
                [e_v, h_v] = voidNearestSurfaceExclusion(...
                    self.microstructData{n}.diskCenters,...
                    self.microstructData{n}.diskRadii,...
                    gridRF, dist);
                self.designMatrix{n} = [self.designMatrix{n}, e_v(:), h_v];
            end
            
            
            %mean pore chord length non-overlap. polydis. spheres
            for n = 1:numel(self.samples)
                phi = meanChordLength(self.microstructData{n}.diskCenters,...
                    self.microstructData{n}.diskRadii, gridRF);
                self.designMatrix{n} = [self.designMatrix{n}, phi(:)];
            end
            
            %log mean pore chord length non-overlap. polydis. spheres
            for n = 1:numel(self.samples)
                phi =log(meanChordLength(self.microstructData{n}.diskCenters,...
                    self.microstructData{n}.diskRadii, gridRF) + eps);
                self.designMatrix{n} = [self.designMatrix{n}, phi(:)];
            end
            
            %exp mean pore chord length non-overlap. polydis. spheres
            for n = 1:numel(self.samples)
                phi =exp(meanChordLength(self.microstructData{n}.diskCenters,...
                    self.microstructData{n}.diskRadii, gridRF));
                self.designMatrix{n} = [self.designMatrix{n}, phi(:)];
            end
            
            %sqrt mean pore chord length non-overlap. polydis. spheres
            for n = 1:numel(self.samples)
                phi=sqrt(meanChordLength(self.microstructData{n}.diskCenters,...
                    self.microstructData{n}.diskRadii, gridRF));
                self.designMatrix{n} = [self.designMatrix{n}, phi(:)];
            end
            
            %2-point correlation
            dist = .1;
            for n = 1:numel(self.samples)
                phi= twoPointCorrelation(self.microstructData{n}.diskCenters,...
                    self.microstructData{n}.diskRadii, gridRF, true, dist);
                self.designMatrix{n} = [self.designMatrix{n}, phi(:)];
            end
            dist = .075;
            for n = 1:numel(self.samples)
                phi= twoPointCorrelation(self.microstructData{n}.diskCenters,...
                    self.microstructData{n}.diskRadii, gridRF, true, dist);
                self.designMatrix{n} = [self.designMatrix{n}, phi(:)];
            end
            dist = .05;
            for n = 1:numel(self.samples)
                phi= twoPointCorrelation(self.microstructData{n}.diskCenters,...
                    self.microstructData{n}.diskRadii, gridRF, true, dist);
                self.designMatrix{n} = [self.designMatrix{n}, phi(:)];
            end
            dist = .025;
            for n = 1:numel(self.samples)
                phi= twoPointCorrelation(self.microstructData{n}.diskCenters,...
                    self.microstructData{n}.diskRadii, gridRF, true, dist);
                self.designMatrix{n} = [self.designMatrix{n}, phi(:)];
            end
            
            
            %log 2-point correlation
            dist = .1;
            for n = 1:numel(self.samples)
                phi = log(twoPointCorrelation(...
                    self.microstructData{n}.diskCenters,...
                    self.microstructData{n}.diskRadii, gridRF, true, dist)+eps);
                self.designMatrix{n} = [self.designMatrix{n}, phi(:)];
            end
            dist = .075;
            for n = 1:numel(self.samples)
                phi = log(twoPointCorrelation(...
                    self.microstructData{n}.diskCenters,...
                    self.microstructData{n}.diskRadii, gridRF, true, dist)+eps);
                self.designMatrix{n} = [self.designMatrix{n}, phi(:)];
            end
            dist = .05;
            for n = 1:numel(self.samples)
                phi = log(twoPointCorrelation(...
                    self.microstructData{n}.diskCenters,...
                    self.microstructData{n}.diskRadii, gridRF, true, dist)+eps);
                self.designMatrix{n} = [self.designMatrix{n}, phi(:)];
            end
            dist = .025;
            for n = 1:numel(self.samples)
                phi = log(twoPointCorrelation(...
                    self.microstructData{n}.diskCenters,...
                    self.microstructData{n}.diskRadii, gridRF, true, dist)+eps);
                self.designMatrix{n} = [self.designMatrix{n}, phi(:)];
            end
            
            
            %sqrt 2-point correlation
            dist = .1;
            for n = 1:numel(self.samples)
                phi = sqrt(twoPointCorrelation(...
                    self.microstructData{n}.diskCenters,...
                    self.microstructData{n}.diskRadii, gridRF, true, dist)+eps);
                self.designMatrix{n} = [self.designMatrix{n}, phi(:)];
            end
            dist = .075;
            for n = 1:numel(self.samples)
                phi = sqrt(twoPointCorrelation(...
                    self.microstructData{n}.diskCenters,...
                    self.microstructData{n}.diskRadii, gridRF, true, dist)+eps);
                self.designMatrix{n} = [self.designMatrix{n}, phi(:)];
            end
            dist = .05;
            for n = 1:numel(self.samples)
                phi = sqrt(twoPointCorrelation(...
                    self.microstructData{n}.diskCenters,...
                    self.microstructData{n}.diskRadii, gridRF, true, dist)+eps);
                self.designMatrix{n} = [self.designMatrix{n}, phi(:)];
            end
            dist = .025;
            for n = 1:numel(self.samples)
                phi = sqrt(twoPointCorrelation(...
                    self.microstructData{n}.diskCenters,...
                    self.microstructData{n}.diskRadii, gridRF, true, dist)+eps);
                self.designMatrix{n} = [self.designMatrix{n}, phi(:)];
            end
            
        end
        
        function shapeToLocalDesignMat(self)
            %Sets separate coefficients theta_c for each macro-cell in a single
            %microstructure sample. Don't execute before rescaling/
            %standardization of design Matrix!
            debug = false; %debug mode
            disp(strcat('Using separate feature coefficients theta_c for', ...
                ' each macro-cell in a microstructure...'));
            [nElc, nFeatureFunctions] = size(self.designMatrix{1});
            Phi{1} = zeros(nElc, nElc*nFeatureFunctions);
            nData = numel(self.designMatrix);
            Phi = repmat(Phi, nData, 1);
            
            %Reassemble design matrix
            for n = 1:nData
                for k = 1:nElc
                    Phi{n}(k, ((k - 1)*nFeatureFunctions + 1):...
                        (k*nFeatureFunctions)) = self.designMatrix{n}(k, :);
                end
                Phi{n} = sparse(Phi{n});
            end
            if debug
                firstDesignMatrixBeforeLocal = self.designMatrix{1}
                firstDesignMatrixAfterLocal = full(Phi{1})
                pause
            end
            self.designMatrix = Phi;
            disp('done')
        end
        
        function [featFuncMin, featFuncMax] = computeFeatureFunctionMinMax(self)
            %Computes min/max of feature function outputs over training data,
            %separately for every macro cell
            featFuncMin = self.designMatrix{1};
            featFuncMax = self.designMatrix{1};
            for n = 1:numel(self.designMatrix)
                featFuncMin(featFuncMin > self.designMatrix{n}) =...
                    self.designMatrix{n}(featFuncMin > self.designMatrix{n});
                featFuncMax(featFuncMax < self.designMatrix{n}) =...
                    self.designMatrix{n}(featFuncMax < self.designMatrix{n});
            end
        end
        
        function rescaleDesignMatrix(self, featFuncMin, featFuncMax)
            %Rescale design matrix s.t. outputs are between 0 and 1
            disp('Rescale design matrix...')
            
            if nargin < 3
                [featFuncMin, featFuncMax] = self.computeFeatureFunctionMinMax;
            end
            
            featFuncDiff = featFuncMax - featFuncMin;
            %to avoid irregularities due to rescaling (if every macro cell
            %has the same feature function output). Like this rescaling does
            %not have any effect
            featFuncMin(featFuncDiff == 0) = 0;
            featFuncDiff(featFuncDiff == 0) = 1;
            for n = 1:numel(self.designMatrix)
                self.designMatrix{n} =...
                    (self.designMatrix{n} - featFuncMin)./(featFuncDiff);
            end
            
            %Check if design Matrices are real and finite
            self.checkDesignMatrices;
            self.saveNormalization('rescaling', featFuncMin, featFuncMax);
            disp('done')
        end
        
        function checkDesignMatrices(self)
            %Check for finiteness
            for n = 1:numel(self.designMatrix)
                if(~all(all(all(isfinite(self.designMatrix{n})))))
                    dataPoint = n
                    Phi = self.designMatrix{n}
                    pause
                    [coarseElement, featureFunction] = ...
                        ind2sub(size(self.designMatrix{n}),...
                        find(~isfinite(self.designMatrix{n})))
                    warning(strcat('Non-finite design matrix.',...
                        ' Setting non-finite component to 0.'))
                    self.designMatrix{n}(~isfinite(self.designMatrix{n})) = 0;
                elseif(~all(all(all(isreal(self.designMatrix{n})))))
                    warning('Complex feature function output:')
                    dataPoint = n
                    Phi = self.designMatrix{n}
                    pause
                    [coarseElement, featureFunction] =...
                        ind2sub(size(self.designMatrix{n}),...
                        find(imag(self.designMatrix{n})))
                    disp('Ignoring imaginary part...')
                    self.designMatrix{n} = real(self.designMatrix{n});
                end
            end
        end
        
        function saveNormalization(self, type, a, b)
            disp('Saving design matrix normalization...')

            if ~exist('./data')
                mkdir('./data');
            end
            if strcmp(type, 'standardization')
                save('./data/featureFunctionMean', 'a', '-ascii');
                save('./data/featureFunctionSqMean', 'b', '-ascii');
            elseif strcmp(type, 'rescaling')
                save('./data/featureFunctionMin', 'a', '-ascii');
                save('./data/featureFunctionMax', 'b', '-ascii');
            else
                error('Which type of data normalization?')
            end
        end
        
        function [triHandles, pltHandles, figHandle, cb] =...
                plot(self, samples)
            %Plots the fine scale data and returns handles
                        
            %Load data to make sure that not interpolated data is plotted
            self.readData('c');
            self.readData('x');
            self.readData('p');
            self.readData('u');
            
            figHandle = figure;
            pltIndex = 1;
            N = numel(samples);
            for n = samples
                figure(figHandle);
                %pressure field
                pltHandles(1, pltIndex) = subplot(3, N, pltIndex);
                triHandles(1, pltIndex) =...
                    trisurf(self.cells{n}, self.X{n}(:, 1),...
                    self.X{n}(:, 2), self.P{n});
                triHandles(1, pltIndex).LineStyle = 'none';
                axis square;
                axis tight;
                view(3);
                grid off;
                box on;
                xticks({});
                yticks({});
                cb(1, pltIndex) = colorbar;
                cb(1, pltIndex).Label.String = 'pressure p';
                cb(1, pltIndex).Label.Interpreter = 'latex';
                
                %velocity field (norm)
                u_norm = sqrt(sum(self.U{n}.^2));
                pltHandles(2, pltIndex) = subplot(3, N, pltIndex + N);
                triHandles(2, pltIndex) = trisurf(self.cells{n},...
                   self.X{n}(:, 1), self.X{n}(:, 2), u_norm);
                triHandles(2, pltIndex).LineStyle = 'none';
                axis square;
                axis tight;
                view(3);
                grid off;
                box on;
                xticks({});
                yticks({});
                cb(2, pltIndex) = colorbar;
                cb(2, pltIndex).Label.String = 'velocity norm $|u|$';
                cb(2, pltIndex).Label.Interpreter = 'latex';
                
                %velocity field (norm), 2d
                pltHandles(3, pltIndex) = subplot(3, N, pltIndex + 2*N);
                triHandles(3, pltIndex) = trisurf(self.cells{n},...
                   self.X{n}(:, 1), self.X{n}(:, 2), u_norm);
                triHandles(3, pltIndex).LineStyle = 'none';
                axis square;
                axis tight;
                view(2);
                grid off;
                box on;
                xticks({});
                yticks({});
                cb(3, pltIndex) = colorbar;
                cb(3, pltIndex).Label.String = 'velocity norm $|u|$';
                cb(3, pltIndex).Label.Interpreter = 'latex';
                pltIndex = pltIndex + 1;
            end
        end
    end
end

