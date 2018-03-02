function [outp_array] = blockMean(inp_array, nr, nc)

outp_array = zeros(nr, nc);
[Nr, Nc] = size(inp_array);
lr = ceil(Nr/nr);
lc = ceil(Nc/nc);
for r = 1:nr
    end_r = min(r*lr, Nr);
    for c = 1:nc
        end_c = min(c*lc, Nc);
        outp_array(r, c) = mean2(inp_array(((r - 1)*lr + 1):end_r, ...
            ((c - 1)*lc + 1):end_c));
    end
end

end

