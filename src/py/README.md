NOTES
======

A typical execution sequence:
-----------------------------

1. Take a problem, ex. debmdk
2. Generate the data set by executing the file `genpf.py`, edit the `main()` accordingly before doing that.
2. A file like `debmdk3m500n.csv` will be generated. 
3. Now, execute `calcmu.py`, it will compute the knee indicator values (mu), this will generate a file `debmdk3m500n-mu.csv`
4. Use the `plotknees.py` script to plot `debmdk3m500n-mu.csv` file.
5. Now use `tsne.py` script to calculate the TSNE coordinate values, this operation will generate `debmdk3m500n-tsne.csv` file.
6. Use the `plottsne.py` script to plot `debmdk3m500n-tsne.csv` file. 
7. To calculate the knee indicator values (mu) from the TSNE coordinate use `wsntsne.py` script. This operatio will generate `debmdk3m500n-wsntsne.csv` file.
8. You can again use the `plottsne.py` script to plot the `debmdk3m500n-wsntsne.csv` file.
9. All the previous knee indicator value calculator computes a single value, we have another version of knee indicator that represents the knee values as a vector of numbers. You can use the `muvectsne.py` script to compute these values. The input file will be `debmdk3m500n-tsne.csv` and the output will be `debmdk3m500n-muvtsne.csv`. 

---------------------------------------------------

**1. wsn does not work:**
	if there are other points closer to a principal knee, sometimes those point recieves 0.0 
	indicator values.
 
		reason: the wsn metric considers the minimum possible trade-off loss/gain ratio, 
			therefore a very close point to the real knee gets 0.0 indicator value.

		solution: instead of taking min, take median, this solves the above problem

		problem: but now, wsn indicator disregard the CHIM points as knees and also detects 
			concave fronts as knees.

**2. use tsne neighbourhood for wsn:**
	looks like this idea is somewhat working, may be we need to some tuning regarding the tsne
	parameters. The good thing with approach is that we do not need to change the neighbourhood
	radius according to the dimension. A constant epsilon seems to work for 3, 4, and 5 dim single
	knee PF.

**3. try with the trade-off vector:**
	instead of actual obj values, use the trade-off vector for tsne, turns out that it does not
	work as expected

**4. dtlz6:**
	the patches are not intact in the tsne 

---------------------------------------------------

**1. dtlz6:** 
	patches are somewhat clear up to 4 dimensions, then it gets unintelligible.

**2. trade-off vector and objective value for tsne:** 
	does not give any interesting output, see the tsne analysis

**3. plot points with varying size:**
	done 

**5. radviz:**
	completely useless in higher dimension

**6. scatterplot matrix:**
	no extra information abouth the isolated patches is visible

**4. dtlz7:**

**5. outlier problem:**
	similar results to dtlz6

**6. detailed analysis on tsne:**
	do not assume tsne give you pattern or structure, tsne just gives you one configuration
	of the original space provided that the neighbourhood information are preserved.
	
**7. cluster and color the patches for tsne plotting:**
	the clustering algorithm DBSCAN does not scale good for higher dimension, DBSCAN
	is the best algorithm known so far.

**8.** wsn is greatly improved after normalizing by (max(f) - min(f)), this will put smaller mu values
	to the extreme solutions.

**9. make a function with no concave end points:**
	done, got better result with the normalized mu values.


**10. test wsn with tsne neighbourhood for 10-d:**
  * debmdk:
  * dtlz6:
  * dtlz7:
  * outlier:

**11. n-dimensional boundary finding algorithm:**
	make a n-dimensional sphere and apply tsne and see if the boundaries are
	separated.


		

