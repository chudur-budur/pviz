% create data on a m-dim sphere on the positive coordinate
clear all;  
hold on;
% f = layerwise_Deb(4, 1, [1], [13]);
% f = f';
M = 4;
x = rand(3,2000);
f(1,:) = prod(cos(pi/2*x(1:M-1,:)),1);
for ii = 2:M-1
   f(ii,:) = prod(cos(pi/2*x(1:M-ii,:)),1) .* ...
      sin(pi/2*x(M-ii+1,:));
end
f(M,:) = sin(pi/2*x(1,:));

%f = importdata('dataKhaled.out');
%f = f';
%plot3(f(1,:),f(2,:),f(3,:),'o');
[M,N] = size(f),
u = ones(M,1)/sqrt(M);
for i=1:N
    fp(:,i) = f(:,i) - (u'*f(:,i))*u + u/sqrt(M);
end
%plot3(fp(1,:),fp(2,:),fp(3,:),'ro'); hold on;
% boundary seeking algorithm
% calculate centroid of fp
angle_thres = 30.0; % in degrees
cen = mean(fp,2);
% plot3(cen(1),cen(2),cen(3),'x');
for i=1:N
    dist(i) = (sum(fp(:,i)-cen).^2)^0.5;
end
[~,id] = sort(dist,'descend');
flag = zeros(1,N);
ic = 1;
considered = 0;
count_considered = [];
while ((ic<=2) && (considered < N))
    for i=1:N
        if (flag(id(i)) == 0)
            pivot = fp(:,id(i));
            unitvec = (pivot-cen)/sqrt(sum((pivot-cen).^2));
            flagi = 0;
            for j=1:N
                if ((flag(j) ~= -2) && (j ~= id(i)))
                    fpvec = (fp(:,j)-pivot)/sqrt(sum((fp(:,j)-pivot).^2));
                    angle = (180.0/pi)*acos(fpvec'*unitvec);
                    if ((angle < angle_thres) && (angle >= 0.0))
                        flag(id(i)) = -1; % pt i cannot be a boundary pt
                        flagi = 1;
                    end
                end
            end
            if (flagi == 0)
                flag(id(i)) = ic; % pt i is a boundary pt
            end
        end
    end
    bound = [];
    for i=1:N
        if (flag(id(i)) == ic)
            flag(id(i)) = -2;
            bound = [bound; f(:,id(i))'];
        end
        if flag(id(i)) == -1
            flag(id(i)) = 0;
        end
    end
    considered = sum(flag ~= 0);
    count_considered = [count_considered, [considered]];
    if size(bound,1) > 0
        radviz3(bound, 1.0/ic);
    end
    ic = ic + 1;
end
count_considered,
hold off