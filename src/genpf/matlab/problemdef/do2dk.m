function [parent_pop] = do2dk(parent_pop)
%   This procedure implements do2dk function.
%   The canonical zdt1 function is defined as below --
%   f_1 = x_1
%   f_2 = g * (1.0 - sqrt(f_1/g))
%   g(x_2, x_3, ..., x_n) = 1.0 + (9/(n - 1)) sum_{i = 2}^n x_i
%   0 <= x_i <= 1.0 (i = 1, 2, 3, ..., n)

global nreal ;

K = 3 ;
s = 1.0 ;
x = parent_pop(:,1:nreal);
g = 1 + ((9 / (nreal - 1)) .* sum(x(:,2:nreal), 2));
r = 5 + (10 .* ((x(:,1) - 0.5).^2)) ...
        + ((1/K) .* cos((2 * K * pi) .* x(:,1)) .* 2^(s / 2));
f1 = g .* r .* ((sin((pi .* x(:,1) ./ (2 ^ (s + 1))) ...
        + (1 + ((2^s - 1)/(2^(s + 2))) * pi))) + 1);
f2 = g .* r .* ((cos((pi .* (x(:,1) ./ 2)) + pi)) + 1);
parent_pop(:, (nreal+1)) = f1;
parent_pop(:, (nreal+2)) = f2;

end
