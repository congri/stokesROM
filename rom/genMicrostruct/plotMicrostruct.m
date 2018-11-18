function [img_handle, fig_handle] =...
    plotMicrostruct(diskCenters, diskRadii, resolution)
%Plots a microstructure of polydisperse spherical exclusions based on
%disk centers and radii
if nargin < 3
    resolution = 2048;
end

% tic;
% [xx, yy] = meshgrid(linspace(0, 1, resolution));
% x = [xx(:) yy(:)];
% clear xx yy;
% 
% img = true(resolution);
% 
% %loop over every pixel and check if solid or fluid
% for i = 1:size(x, 1)
%     distSq = sum((diskCenters - x(i, :)).^2, 2);
%     if any(distSq' <= diskRadii.^2)
%         %inside of exclusion, i.e. outside of domain
%         img(i) = false;
%     end
% end
% t1 = toc
% 
% fig_handle = figure;
% img_handle = imagesc(img);
% grid off;
% xticks([]);
% yticks([]);
% 
% tic;
% %built-in method
% figure;
% visc = viscircles(diskCenters, diskRadii)
% t2 = toc


%loop over cirlces instead
[xx, yy] = meshgrid(linspace(0, 1, resolution));
r2 = diskRadii.^2;
img = false(resolution);

for n = 1:numel(diskRadii)
    img = img | ((xx - diskCenters(n, 1)).^2 + (yy - diskCenters(n, 2)).^2 ...
        <= r2(n));
end

fig_handle = figure;
img_handle = imagesc(img);
grid off;
xticks([]);
yticks([]);
ax = gca;
colormap(ax, gray);
axis square;
end

