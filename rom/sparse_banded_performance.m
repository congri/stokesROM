load('./stiffness');
load('./force');

N = 1e4;
K = sparse(K);
size_K = size(K)


tic
for n = 1:N
    u = K\f;
end
t = toc/N
disp('s per solution')