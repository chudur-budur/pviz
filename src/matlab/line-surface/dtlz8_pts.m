clear all;
C  = [];
% M = 3; N = 6000;
% M = 4; N = 150000;
% M = 6; N = 3000;
M = 8; N = 4000;
pt1 = zeros(1, M); pt1(M) = 1.0;
pt2 = ones(1, M) / (M + 1);
B = ones(101,1) * pt1 + [0:0.01:1]' * (pt2 - pt1); % 101 pts on line
C = ones(N-101, M-1) / (M + 1) + ...
    lhsdesign(N - 101, M - 1) * (1 - 1.0 / (M + 1));
D = [];
for i = 1:(N - 101)
    C(i,M) = (1 - sum(C(i,:))) / 2.0;
    % sm = sum(C(i, 1:M - 1)) + 2 * C(i, M);
    % C(i,:) = C(i,:) / sm;
    flag = 0;
    for j = 1:(M - 1)
        if C(i,M) < 1.0 - M * C(i,j)
            flag = 1;
            break;
        end
    end
    if (flag == 0) % && C(i,M) >= 0)
        D = [D;C(i,:)];
    end
end

C = [B;D];
[A,id] = prtp(C);

%parallelcoords(A);

%T = clusterdata(A(:,1:2),'maxclust',2);
%scatter3(A(:,1),A(:,2),A(:,3),20,T);

plot3(A(:,1),A(:,2),A(:,3),'o');
xlabel('f1');ylabel('f2');zlabel('f3');
hold off;

dlmwrite(strcat('line-', num2str(size(A,2)), 'd.out'), A, ...
    'delimiter', '\t', 'precision', '%e', 'newline', 'unix');