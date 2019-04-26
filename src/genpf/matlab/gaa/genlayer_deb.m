% This function is written by Dr. Aimin Zhou and modified by K. Deb for generating any number of 
% weight vectors. "layers" is # of layers, reduc is a vector of reduction
% factor for each layer, and gap is a vector gaps in each layer. 
% For example, run 
% a = layerwise_Deb(3, 3, [1,0.6,0.2], [1,3,1])
% a = layerwise_Deb(4, 4, [1,0.75,0.4,0.1], [2,3,2,1]); %my suggestion, change

function Wt = genlayer_deb(objDim, layers, reduc, gap)

    Wt = [];
    
    for i = 1 : layers
        N = factorial(objDim + gap(i) - 1) / ...
            (factorial(gap(i)) * factorial(objDim-1));
        diff = (1.0 - reduc(i)) / objDim;
        W1 = reduc(i) * initweight(objDim, N) + repmat(diff, objDim, N);
        Wt = [Wt, W1];
    end
    Wt = Wt';
end
        
function W = initweight(objDim, N)
    
    U = floor(N ^ (1 / (objDim - 1))) - 2;
    M = 0;
    while M < N
        U = U + 1;
        M = noweight(U, 0, objDim); 
    end

    W = zeros(objDim, M);
    C = 0;
    V = zeros(objDim, 1);
    [W, C] = setweight(W, C, V, U, 0, objDim, objDim);
    W = W / (U + 0.0);

    pos = (W < 1.0E-5);
    W(pos) = 1.0E-5;
end

%%
function M = noweight(unit, sum, dim)

    M = 0;

    if dim == 1
        M = 1; 
        return;
    end

    for i = 0 : 1 : (unit - sum)
        M = M + noweight(unit, sum + i, dim - 1);
    end

end

%%
function [w, c] = setweight(w, c, v, unit, sum, objdim, dim)

    if dim == objdim
        v = zeros(objdim, 1);
    end

    if dim == 1
        c = c + 1;
        v(1) = unit - sum;
        w(:, c) = v;
        return;
    end

    for i = 0 : 1 : (unit - sum)
        v(dim) = i;
        [w, c] = setweight(w, c, v, unit, sum + i, objdim, dim - 1);
    end
end