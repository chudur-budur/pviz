NOTES
======

1. wsn does not work:
	if there are other points closer to a principal knee, sometimes those point recieves 0.0 
	indicator values.
 
		reason: the wsn metric considers the minimum possible trade-off loss/gain ratio, 
			therefore a very close point to the real knee gets 0.0 indicator value.

		solution: instead of taking min, take median, this solves the above problem

		problem: but now, wsn indicator disregard the CHIM points as knees and also detects 
			concave fronts as knees.

2. use tsne neighbourhood for wsn:

	looks like this idea is somewhat working, may be we need to some tuning regarding the tsne
	parameters. The good thing with approach is that we do not need to change the neighbourhood
	radius according to the dimension. A constant epsilon seems to work for 3, 4, and 5 dim single
	knee PF.

3. try with the trade-off vector:
	instead of actual obj values, use the trade-off vector for tsne, turns out that it does not
	work as expected

4. dtlz6:
	the patches are not intact in the tsne 

---------------------------------------------------

1. dtlz6:
	patches are somewhat clear up to 4 dimensions, then it gets unintelligible.

2. trade-off vector and objective value for tsne:
	does not give any interesting output, see the tsne analysis

3. plot points with varying size:
	done 

5. radviz:
	completely useless in higher dimension

6. scatterplot matrix:
	no extra information abouth the isolated patches is visible

4. dtlz7:

5. outlier problem:
	similar results to dtlz6

6. detailed analysis on tsne:
	do not assume tsne give you pattern or structure, tsne just gives you one configuration
	of the original space provided that the neighbourhood information are preserved.
	
7. cluster and color the patches for tsne plotting:
	the clustering algorithm DBSCAN does not scale good for higher dimension, DBSCAN
	is the best algorithm known so far.

8. wsn is greatly improved after normalizing by (max(f) - min(f)), this will put smaller mu values
	to the extreme solutions.

9. make a function with no concave end points:
	done, got better result with the normalized mu values.


10. test wsn with tsne neighbourhood for 10-d:
	debmdk:
	dtlz6:
	dtlz7:
	outlier:

11. n-dimensional boundary finding algorithm:
	make a n-dimensional sphere and apply tsne and see if the boundaries are
	separated.


		

