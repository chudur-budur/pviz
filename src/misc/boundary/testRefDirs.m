objCount = 4;
pointCount = 500;
filename = strcat('refdir', num2str(objCount), 'm', num2str(pointCount), 'n.txt')
% fp = fopen(filename, 'w')
D = initweight(objCount, pointCount)'
if objCount == 2
    scatter(D(:,1), D(:,2));
    % fprintf(fp, '%f %f\n', D(:,1), D(:,2))
elseif objCount == 3
    scatter3(D(:,1), D(:,2), D(:,3));
    % fprintf(fp, '%f %f %f\n', D(:, 1), D(:, 2), D(:, 3))
end
% fp.close()


