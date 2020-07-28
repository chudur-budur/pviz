MATLAB code to generate Pareto-front for the GAA problem
=========================================================

1. `save_layer.m`: this script will save 3112 reference directions using Das-Dennis's method. The saved file is `weights-layer-3112.mat`. The number of reference directions can be changed by setting appropriate layer parameters.
2. `save_lhs.m`: this script will save 3112 reference directions using LHS. The saved file is `weights-lhs-3112.mat`. Just change the number the points to change the number of reference directions.
3. `gaa_das_solver.m`: this script will solve the GAA problem using reference directions from `weight-layer-3112.m`. The output files are `gaa-das-10d.out/.mat` (objective function values), `gaa-das-10d-g.out/.mat` (individual constraint values), `gaa-das-10d-cv.out/.mat` (cumulative constraint violation (cv), i.e. cv <= 0: feasible and cv > 0: infeasible).
    * `gaa_das_parallel_solver.m`: this script will solve the GAA problem using parallel for loop to make everything faster.
    * `gaa_das_parallel_solver.sb`: this is the slurm submission script (over HPCC cluster) for the parallel for loop code.
4. `gaa_lhs_solver.m`: this script will solve the GAA problem using reference directions from `weight-lhs-3112.m`. The output files are `gaa-lhs-10d.out/.mat` (objective function values), `gaa-lhs-10d-g.out/.mat` (individual constraint values), `gaa-lhs-10d-cv.out/.mat` (cumulative constraint violation (cv), i.e. cv <= 0: feasible and cv > 0: infeasible).
    * `gaa_lhs_parallel_solver.m`: this script will solve the GAA problem using parallel for loop to make everything faster.
    * `gaa_lhs_parallel_solver.sb`: this is the slurm submission script (over HPCC cluster) for the parallel for loop code.
