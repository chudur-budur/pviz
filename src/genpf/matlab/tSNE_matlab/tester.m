% load 'mnist_train.mat'
load 'car-crash-clusters'
load 'car-crash-boundary'

global max_iter;

rng(1000);

[~,bidx] = intersect(xtrain, boundary, 'rows');
bidx
xtrain(bidx,:)
xlabels(bidx,:)

%numrows = size(xtrain, 1);
%ind = randperm(numrows);
%xtrain = xtrain(ind(1:numrows),:);
%xlabels = xlabels(ind(1:numrows));



no_dims = 2;
initial_dims = 3;
perplexity = 30;
max_iter = 500;

yvals = tsne(xtrain, [], no_dims, initial_dims, perplexity);

% plot cluster 0
box on;
scatter(yvals(xlabels == 0, 1), yvals(xlabels == 0, 2), 20, 'k', 'd');
hold on;
% plot cluster 1
box on;
scatter(yvals(xlabels == 1, 1), yvals(xlabels == 1, 2), 20, 'k', '+');
hold on;
% plot cluster 2
box on;
scatter(yvals(xlabels == 2, 1), yvals(xlabels == 2, 2), 20, 'k', 'o');
hold on;
% plot boundaries
box on;
scatter(yvals(bidx,1), yvals(bidx,2), 170, 'k', '^', 'filled');
hold off;