%inversion performance
clear;
rng('shuffle');
n_iter = 20;
nElc = 64;
nFeatures = 200;
for k = 1:nElc
    L = normrnd(0, 1, nFeatures);
    A(((k-1)*nFeatures + 1):(k*nFeatures),...
        ((k - 1)*nFeatures + 1):(k*nFeatures)) = L'*L;
end

% tic
% for i = 1:n_iter
%     tmp = invChol_mex(A);
% end
% t1 = toc
% 
% tic
% for i = 1:n_iter
%     tmp = inv(A);
% end
% t2 = toc
% 
% I = eye(nElc*nFeatures);
% tic
% for i = 1:n_iter
%     tmp = A\I;
% end
% t3 = toc
% 
% A = sparse(A);
% I = speye(nElc*nFeatures);
% tic
% for i = 1:n_iter
%     tmp = A\I;
% end
% t4 = toc


%Exploit block structure
tmp = zeros(nElc*nFeatures);
tic
for i = 1:n_iter
    for k = 1:nElc
        tmp(((k-1)*nFeatures + 1):(k*nFeatures),...
            ((k - 1)*nFeatures + 1):(k*nFeatures)) = invChol_mex(A(((k-1)*nFeatures + 1):(k*nFeatures),...
        ((k - 1)*nFeatures + 1):(k*nFeatures)));
    end
end
t5 = toc

tic
for i = 1:n_iter
    for k = 1:nElc
        tmp(((k-1)*nFeatures + 1):(k*nFeatures),...
            ((k - 1)*nFeatures + 1):(k*nFeatures)) = inv(A(((k-1)*nFeatures + 1):(k*nFeatures),...
        ((k - 1)*nFeatures + 1):(k*nFeatures)));
    end
end
t6 = toc

I = speye(nFeatures);
tic
for i = 1:n_iter
    for k = 1:nElc
        tmp(((k-1)*nFeatures + 1):(k*nFeatures),...
            ((k - 1)*nFeatures + 1):(k*nFeatures)) = A(((k-1)*nFeatures + 1):(k*nFeatures),...
        ((k - 1)*nFeatures + 1):(k*nFeatures))\I;
    end
end
t7 = toc

A = sparse(A);
I = speye(nFeatures);
tic
for i = 1:n_iter
    for k = 1:nElc
        tmp(((k-1)*nFeatures + 1):(k*nFeatures),...
            ((k - 1)*nFeatures + 1):(k*nFeatures)) = A(((k-1)*nFeatures + 1):(k*nFeatures),...
        ((k - 1)*nFeatures + 1):(k*nFeatures))\I;
    end
end
t8 = toc


%Exploit parallelization
parPoolInit(nElc);
clear tmp;
tmp{1} = zeros(nFeatures);
tmp = repmat(tmp, nElc, 1);
AA = tmp;
for k = 1:nElc
    AA{k} = A(((k-1)*nFeatures + 1):(k*nFeatures),...
        ((k - 1)*nFeatures + 1):(k*nFeatures));
end
%invert block diagonal matrix in parallel
tic
for i = 1:n_iter
    parfor k = 1:nElc
        tmp{k} = invChol_mex(AA{k});
    end
end
t5 = toc
