function radviz3(fdata,level);
% plot a circle
xc = [];
for i=0:100
    ang = i*2.0*pi/100.0;
    xc = [xc; [cos(ang) sin(ang)]];
end
% fdata = [
%          0.0 1.0 0.0 0 0;
%          0.1 0.9 0.0 0 0;
%          0.2 0.8 0.0 0 0;
%          0.3 0.7 0.0 0 0;
%          0.4 0.6 0.0 0 0;
%          0.5 0.5 0.0 0 0;
%          0.6 0.4 0.0 0 0;
%          0.7 0.3 0.0 0 0;
%          0.8 0.2 0.0 0 0;
%          0.9 0.1 0.0 0 0;
%          1.0 0.0 0.0 0 0;
%          0.0 0.95 0.05 0 0;
%          0.1 0.90 0.05 0 0;
%          0.2 0.75 0.05 0 0;
%          0.3 0.65 0.05 0 0;
%          0.4 0.55 0.05 0 0;
%          0.5 0.45 0.05 0 0;
%          0.6 0.35 0.05 0 0;
%          0.7 0.25 0.05 0 0;
%          0.8 0.15 0.05 0 0;
%          0.9 0.05 0.05 0 0
%          ];
[n,m] = size(fdata);   % n = rows, m = cols = #obj
plot3(xc(:,1),xc(:,2),zeros(101,1),'r-'); 
axis equal;
xfix = [];
for i=1:m
    angle = (i-1)*(2*pi)/m;
    xfix = [xfix; [cos(angle) sin(angle)]];
end
x = [];
for j=1:n
    sumf = sum(ones(1,m).*(fdata(j,:)));
    x = [x; [(sum(xfix(:,1)'.*(fdata(j,:)))./sumf)' ...
        (sum(xfix(:,2)'.*(fdata(j,:)))./sumf)']];
end
plot3(xfix(:,1),xfix(:,2),zeros(size(xfix(:,1))),'r*-'); 
xfixfn = [xfix(1,:);xfix(end,:)];
plot3(xfixfn(:,1),xfixfn(:,2),zeros(size(xfixfn(:,1))),'r-');
plot3(x(:,1),x(:,2),ones(size(x(:,1)))*level,'o','MarkerFaceColor','auto'); 
%hold off;
