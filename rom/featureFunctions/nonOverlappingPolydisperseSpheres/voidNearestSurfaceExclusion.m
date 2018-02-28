function [e_v, h_v] = voidNearestSurfaceExclusion(diskCenters, diskRadii,...
    gridRF, distance)
%Mean chord length for non-overlapping polydisp. spheres
%according to Torquato 6.48

meanRadii = zeros(gridRF.nCells, 1);
meanSqRadii = zeros(gridRF.nCells, 1);
A = zeros(gridRF.nCells, 1);
A0 = A;

n = 1;
for cll = gridRF.cells
    if isvalid(cll{1})  %check if cell has been deleted during splitting
        radii_in_n = diskRadii(cll{1}.inside(diskCenters));
        meanRadii(n) = mean(radii_in_n);
        meanSqRadii(n) = mean(radii_in_n.^2);
        A0(n) = cll{1}.surface;
        A(n) = cll{1}.surface - pi*sum(radii_in_n.^2);
        n = n + 1;
    end
end

porefrac = A./A0;
exclfrac = 1 - porefrac;

S = (meanRadii.^2)./meanSqRadii;
a_0 = (1 + exclfrac.*(S - 1))./(porefrac.^2);
a_1 = 1./porefrac;

x = distance./(2*meanRadii);
e_v = porefrac.*exp(-4*exclfrac.*S.*(a_0.*x.^2 + a_1.*x));
if any(~isfinite(e_v))
    %warning('Non-finite value in voidNearestSurfaceExclusion')
    e_v(~isfinite(e_v)) = 1;
end
if nargout > 1
    h_v = 2*((exclfrac.*S)./(meanRadii)).*(2*a_0.*x + a_1).*e_v;
end
if any(~isfinite(h_v))
    %warning('Non-finite value in voidNearestSurfaceExclusion')
    h_v(~isfinite(h_v)) = 0;
end

