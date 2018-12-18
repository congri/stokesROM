function [shortestPath] = shortestPath(lambdak, dir, resolution)
%Gives minimal distance of fluid phase path from left to right (dir == 'x')
%or top to bottom (dir == 'y')
%   lambdak:        binary pixel input. true is exclusion, false is fluid
if nargin < 3
    resolution = 256;
end

%fluid is true, exclusion is false
lambdak = ~lambdak;

shortestPath = Inf;
if(dir == 'x')
    %Check first if there are true elements on the relevant boundaries
    %(left/right for x)
    left = lambdak(:, 1);
    right = lambdak(:, end);
    if(any(left) && any(right))
        %Loop through left boundary and use right boundary as mask for
        %matlab bwgeodesic function
        leftIndex = find(left);
        for i = leftIndex'
            geo = bwdistgeodesic(lambdak, 1, i);
            if(any(isfinite(geo(:, end))))
                shortestPathTemp = min(geo(:, end));
                if(shortestPathTemp < shortestPath)
                    shortestPath = shortestPathTemp;
                    if(~isfinite(shortestPath))
                        i
                        dir
                        geo
                        error('Zero path length of connected path')
                    end
                end
            end
        end
    end
elseif(dir == 'y')
    %Check first if there are true elements on the relevant boundaries 
    %(top/bottom for y)
    top = lambdak(1, :);
    bottom = lambdak(end, :);
    if(any(top) && any(bottom))
        %Loop through upper boundary and use lower boundary as mask for matlab
        %bwgeodesic function
        topIndex = find(top);
        for i = topIndex
            geo = bwdistgeodesic(lambdak, i, 1);
            if(any(isfinite(geo(end, :))))
                shortestPathTemp = min(geo(end, :));
                if(shortestPathTemp < shortestPath)
                    shortestPath = shortestPathTemp;
                    if(~isfinite(shortestPath))
                        i
                        dir
                        geo
                        error('Zero path length of connected path')
                    end
                end
            end
        end
    end
else
    error('which direction?')
end
%the +1 is heuristic -- why is the shortest path sometimes one pixel
%smaller than cell edge length?
shortestPath = (shortestPath + 1)/resolution;
end

