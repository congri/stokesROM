classdef StokesData < handle
    %class for fine scale data of Stokes equation
    
    properties
        %Seldomly changed parameters are to bechanged here
        meshSize = 128
        numberParams = [6.0, 1.0]   %[min, max] pos. number of circ. exclusions
        numberDist = 'logn';
        margins = [0.01, 0.01, 0.01, 0.01]    %[l., u.] margin for imp. phase
        r_params = [-4.5, .8]    %[lo., up.] bound on random blob radius
        coordDist = 'gauss'
        coordDist_mu = '[0.8, 0.3]'   %only for gauss
        coordDist_cov = '[[0.4, 0.0], [0.0, 0.6]]'
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
                        self.P{cellIndex} = file.p';
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
            
            %This is hard-coded here s.t. it is not forgotten in predictions
            %and computing variance of the data
            self.removeSpikes('p', 4);
        end
        
        function shiftData(self, interp, quantity, point, value)
            %shifts observable data by constant, s.t. quantity(point) = value
            if nargin < 3
                quantity = 'p';
            end
            if nargin < 4
                point = [0, 0];
            end
            if nargin < 5
                value = 0;
            end
            %closest point to origin is set to p = 0
            if contains(quantity, 'p')
                for n = 1:numel(self.P)
                    if interp
                        dist = sum((self.X_interp{1} - point).^2, 2);
                    else
                        dist = sum((self.X{n} - point).^2, 2);
                    end
                    p_temp = self.P{n};
                    [~, min_dist_i] = min(dist);
                    p_point = p_temp(min_dist_i);
                    p_temp = p_temp - p_point + value;
                    self.P{n} = p_temp;
                end
            else
                error('shifting only implemented for P')
            end
        end
        
        function removeSpikes(self, quantity, spikeLimit)
            %this actually shouldn't be used. It artificially removes spikes 
            %from the fine  scale response
            
            if contains(quantity, 'p')
                %remove spikes from pressure field
                for n = 1:numel(self.P)
                    sorted_p = sort(self.P{n});
                    n_p = numel(sorted_p);
                    cut = round(.05*n_p);
                    mean_p = mean(sorted_p((1 + cut):(end - cut)));
                    std_p = std(sorted_p((1 + cut):(end - cut)));
                    self.P{n}(self.P{n} > mean_p + spikeLimit*std_p) =...
                        mean_p + spikeLimit*std_p;
                    self.P{n}(self.P{n} < mean_p - spikeLimit*std_p) =...
                        mean_p - spikeLimit*std_p;
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
%                     p_interp = griddata(self.X{n}(:, 1), self.X{n}(:, 2), ...
%                         self.P{n}, xq(:), yq(:), interpolationMode);
                    F = scatteredInterpolant(self.X{n}(:, 1),...
                        self.X{n}(:, 2), self.P{n});
                    p_interp = F(xq(:), yq(:));
                    
                    %replace original data by interpolated data
                    if ~isempty(smoothingParameter)
                        p_interp = reshape(p_interp, numel(fineGridX), ...
                            numel(fineGridY));
                        if boundarySmoothingPixels > 0
                            %only smooth boundary
                            p_temp = imgaussfilt(p_interp,...
                                smoothingParameter, 'Padding', 'symmetric');
                            p_interp(1:boundarySmoothingPixels, :) = ...
                                p_temp(1:boundarySmoothingPixels, :);
                            p_interp((end - boundarySmoothingPixels):end,:)=...
                                p_temp((end - boundarySmoothingPixels):end,:);
                            p_interp(:, 1:boundarySmoothingPixels) = ...
                                p_temp(:, 1:boundarySmoothingPixels);
                            p_interp(:, (end - boundarySmoothingPixels):end)=...
                                p_temp(:, (end - boundarySmoothingPixels):end);
                        else
                            p_interp= imgaussfilt(p_interp,...
                                smoothingParameter, 'Padding', 'symmetric');
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
                fineGridX, fineGridY, interpolationMode, smoothingParameter,...
                boundarySmoothingPixels)
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
                    smoothingParameter, boundarySmoothingPixels);
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
            
            mData = self.microstructData;
            dMat = self.designMatrix;
            parfor n = 1:numel(self.samples)
                %constant 1
                dMat{n} = [dMat{n}, ones(gridRF.nCells, 1)];
                
                %sum of radii moments
                phi = momentPerVolume(mData{n}.diskCenters,...
                    mData{n}.diskRadii, gridRF, .5);
                dMat{n} = [dMat{n}, phi(:)];
                dMat{n} = [dMat{n}, log(phi(:) + eps)];
                
                phi = momentPerVolume(mData{n}.diskCenters,...
                    mData{n}.diskRadii, gridRF, 1.5);
                dMat{n} = [dMat{n}, phi(:)];
                dMat{n} = [dMat{n}, log(phi(:) + eps)];
                
                %pore fraction
                phi = volumeFractionCircExclusions(...
                    mData{n}.diskCenters,...
                    mData{n}.diskRadii, gridRF);
                dMat{n} = [dMat{n}, phi(:)];
                
                %log pore fraction
                dMat{n} = [dMat{n}, log(phi(:) + eps)];
                
                %sqrt pore fraction
                dMat{n} = [dMat{n}, sqrt(phi(:))];
                
                %square pore fraction
                dMat{n} = [dMat{n}, phi(:).^2];
                
                %exp pore fraction
                dMat{n} = [dMat{n}, exp(phi(:))];
                
                %interface area
                phi = interfacePerVolume(...
                    mData{n}.diskCenters,...
                    mData{n}.diskRadii, gridRF);
                dMat{n} = [dMat{n}, phi(:)];
                
                %log interface area
                dMat{n} = [dMat{n}, log(phi(:) + eps)];
                
                %square log interface area
                dMat{n} = [dMat{n}, log(phi(:) + eps).^2];
                
                %cube log interface area
                dMat{n} = [dMat{n}, log(phi(:) + eps).^3];
                
                %log^4 interface area
                dMat{n} = [dMat{n}, log(phi(:) + eps).^4];
                
                %log^1/2 interface area
                dMat{n} = [dMat{n}, abs(log(phi(:) + eps)).^.5];
                
                %log^1/3 interface area
                dMat{n} = [dMat{n}, abs(log(phi(:) + eps)).^(1/3)];
                
                %log^1/4 interface area
                dMat{n} = [dMat{n}, abs(log(phi(:) + eps)).^.25];
                
%                 %next
%                 phi_temp = phi(:);
%                 phi_temp = [phi_temp(2:end); phi_temp(1)];
%                 dMat{n} = [dMat{n}, log(phi_temp(:) + eps)];
%                 
%                 %preceding
%                 phi_temp = phi(:);
%                 phi_temp = [phi_temp(end); phi_temp(1:(end - 1))];
%                 dMat{n} = [dMat{n}, log(phi_temp(:) + eps)];
%                 
%                 %above (N_c = 4)
%                 phi_temp = phi(:);
%                 phi_temp = [phi_temp(5:end); phi_temp(1:4)];
%                 dMat{n} = [dMat{n}, log(phi_temp(:) + eps)];
%                 
%                 %below (N_c = 4)
%                 phi_temp = phi(:);
%                 phi_temp = [phi_temp((end - 3):end); phi_temp(1:(end - 4))];
%                 dMat{n} = [dMat{n}, log(phi_temp(:) + eps)];
%                 
%                 %transpose
%                 phi_temp = reshape(phi(:), 4, 4)';
%                 dMat{n} = [dMat{n}, log(phi_temp(:) + eps)];
%                                 
%                 %flip
%                 phi_temp = flipud(phi(:));
%                 dMat{n} = [dMat{n}, log(phi_temp(:) + eps)];
%                 
%                 %flip transpose
%                 phi_temp = flipud(phi(:));
%                 phi_temp = reshape(phi_temp, 4, 4)';
%                 dMat{n} = [dMat{n}, log(phi_temp(:) + eps)];
                
                
                
                %exp interface area
                dMat{n} = [dMat{n}, exp(phi(:))];
                
                %sqrt interface area
                dMat{n} = [dMat{n}, sqrt(phi(:))];
                
                %square interface area
                dMat{n} = [dMat{n}, phi(:).^2];
                
                %mean distance between disk edges
                phi = diskDistance(mData{n}.diskCenters,...
                    mData{n}.diskRadii, gridRF, 'mean',...
                    'edge2edge');
                dMat{n} = [dMat{n}, phi(:)];
                
                %log mean distance
                dMat{n} = [dMat{n}, log(phi(:) + eps)];
                
                %square log mean distance
                dMat{n} = [dMat{n}, log(phi(:) + eps).^2];
                
                %flow through thin plates
                %log^3 mean distance
                dMat{n} = [dMat{n}, log(phi(:) + eps).^3];
                
                %Hagen-Poiseuille?
                %log^4 mean distance
                dMat{n} = [dMat{n}, log(phi(:) + eps).^4];
                
                %mean distance between disk centers
                phi = diskDistance(mData{n}.diskCenters,...
                    mData{n}.diskRadii, gridRF, 'mean', 2);
                dMat{n} = [dMat{n}, phi(:)];
                
                
                %min distance between disk centers
                phi = diskDistance(mData{n}.diskCenters,...
                    mData{n}.diskRadii, gridRF, 'min', 2);
                dMat{n} = [dMat{n}, phi(:)];
                
                %log min distance
                dMat{n} = [dMat{n}, log(phi(:) + eps)];
                
                %lin path
                phi = matrixLinealPath(mData{n}.diskCenters,...
                    mData{n}.diskRadii, gridRF, .08);
                dMat{n} = [dMat{n}, phi(:)];
                %log
                dMat{n} = [dMat{n}, log(phi(:) + eps)];
                
                phi = matrixLinealPath(mData{n}.diskCenters,...
                    mData{n}.diskRadii, gridRF, .04);
                dMat{n} = [dMat{n}, phi(:)];
                %log
                dMat{n} = [dMat{n}, log(phi(:) + eps)];
                
                phi = matrixLinealPath(mData{n}.diskCenters,...
                    mData{n}.diskRadii, gridRF, .02);
                dMat{n} = [dMat{n}, phi(:)];
                %log
                dMat{n} = [dMat{n}, log(phi(:) + eps)];
                
                phi = matrixLinealPath(mData{n}.diskCenters,...
                    mData{n}.diskRadii, gridRF, .01);
                dMat{n} = [dMat{n}, phi(:)];
                %log
                dMat{n} = [dMat{n}, log(phi(:) + eps)];
                
                phi = matrixLinealPath(mData{n}.diskCenters,...
                    mData{n}.diskRadii, gridRF, .005);
                dMat{n} = [dMat{n}, phi(:)];
                %log
                dMat{n} = [dMat{n}, log(phi(:) + eps)];
                
                %chord length density
                phi = chordLengthDensity(mData{n}.diskCenters,...
                    mData{n}.diskRadii, gridRF, .04);
                dMat{n} = [dMat{n}, phi(:)];
                %log
                dMat{n} = [dMat{n}, log(phi(:) + eps)];
                
                phi = chordLengthDensity(mData{n}.diskCenters,...
                    mData{n}.diskRadii, gridRF, .02);
                dMat{n} = [dMat{n}, phi(:)];
                %log
                dMat{n} = [dMat{n}, log(phi(:) + eps)];
                
                phi = chordLengthDensity(mData{n}.diskCenters,...
                    mData{n}.diskRadii, gridRF, .01);
                dMat{n} = [dMat{n}, phi(:)];
                %log
                dMat{n} = [dMat{n}, log(phi(:) + eps)];
                
                phi = chordLengthDensity(mData{n}.diskCenters,...
                    mData{n}.diskRadii, gridRF, .005);
                dMat{n} = [dMat{n}, phi(:)];
                %log
                dMat{n} = [dMat{n}, log(phi(:) + eps)];
                
                %nearest surface functions
                [e_v, h_v] = voidNearestSurfaceExclusion(...
                    mData{n}.diskCenters,...
                    mData{n}.diskRadii,...
                    gridRF, .08);
                dMat{n} = [dMat{n}, e_v(:), h_v(:)];
                %log
                dMat{n} = [dMat{n}, log(e_v(:) + eps), log(h_v(:) + eps)];
                
                [e_v, h_v] = voidNearestSurfaceExclusion(...
                    mData{n}.diskCenters,...
                    mData{n}.diskRadii,...
                    gridRF, .04);
                dMat{n} = [dMat{n}, e_v(:), h_v(:)];
                %log
                dMat{n} = [dMat{n}, log(e_v(:) + eps), log(h_v(:) + eps)];
                
                [e_v, h_v] = voidNearestSurfaceExclusion(...
                    mData{n}.diskCenters,...
                    mData{n}.diskRadii,...
                    gridRF, .02);
                dMat{n} = [dMat{n}, e_v(:), h_v(:)];
                %log
                dMat{n} = [dMat{n}, log(e_v(:) + eps), log(h_v(:) + eps)];
                
                [e_v, h_v] = voidNearestSurfaceExclusion(...
                    mData{n}.diskCenters,...
                    mData{n}.diskRadii,...
                    gridRF, .01);
                dMat{n} = [dMat{n}, e_v(:), h_v(:)];
                %log
                dMat{n} = [dMat{n}, log(e_v(:) + eps), log(h_v(:) + eps)];
                
                [e_v, h_v] = voidNearestSurfaceExclusion(...
                    mData{n}.diskCenters,...
                    mData{n}.diskRadii,...
                    gridRF, .005);
                dMat{n} = [dMat{n}, e_v(:), h_v(:)];
                %log
                dMat{n} = [dMat{n}, log(e_v(:) + eps), log(h_v(:) + eps)];
                
                %mean chord length
                phi = meanChordLength(mData{n}.diskCenters,...
                    mData{n}.diskRadii, gridRF);
                dMat{n} = [dMat{n}, phi(:)];
                
                %log mean chord length
                dMat{n} = [dMat{n}, log(phi(:) + eps)];
                
                %exp mean chord length
                dMat{n} = [dMat{n}, exp(phi(:))];
                
                %sqrt mean chord length
                dMat{n} = [dMat{n}, sqrt(phi(:))];
                
                %2-point corr
                phi= twoPointCorrelation(mData{n}.diskCenters,...
                    mData{n}.diskRadii, gridRF, true, .04);
                dMat{n} = [dMat{n}, phi(:)];
                
                %log
                dMat{n} = [dMat{n}, log(phi(:) + eps)];
                
                %sqrt
                dMat{n} = [dMat{n}, sqrt(phi(:))];
                
                %2-point corr
                phi= twoPointCorrelation(mData{n}.diskCenters,...
                    mData{n}.diskRadii, gridRF, true, .02);
                dMat{n} = [dMat{n}, phi(:)];
                
                %log
                dMat{n} = [dMat{n}, log(phi(:) + eps)];
                
                %sqrt
                dMat{n} = [dMat{n}, sqrt(phi(:))];
                
                %2-point corr
                phi= twoPointCorrelation(mData{n}.diskCenters,...
                    mData{n}.diskRadii, gridRF, true, .01);
                dMat{n} = [dMat{n}, phi(:)];
                
                %log
                dMat{n} = [dMat{n}, log(phi(:) + eps)];
                
                %sqrt
                dMat{n} = [dMat{n}, sqrt(phi(:))];
                
                %2-point corr
                phi= twoPointCorrelation(mData{n}.diskCenters,...
                    mData{n}.diskRadii, gridRF, true, .005);
                dMat{n} = [dMat{n}, phi(:)];
                
                %log
                dMat{n} = [dMat{n}, log(phi(:) + eps)];
                
                %sqrt
                dMat{n} = [dMat{n}, sqrt(phi(:))];
            end
            self.designMatrix = dMat;
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

