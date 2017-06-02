#!/usr/local/bin/gnuplot -persist

# data =		"data/debMd_10_190.csv"
# knees =		"data/debMd_10_190-knees.csv"
# aug_knees =		"data/debMd_10_190-aug-knees.csv"

# data =		"data/debMd_3_190.csv"
# knees =		"data/debMd_3_190-knees.csv"
# aug_knees =		"data/debMd_3_190-aug-knees.csv"
# norm_data =		"data/debMd_3_190-norm.csv"
# norm_knees =		"data/debMd_3_190-norm-knees.csv"

# data =		"data/dtlz1_pof_3_190.csv"
# data = 		"data/dtlz2_inv_pof_3_190.csv"
# data = 		"data/dtlz2_mod_inv_pof_3_190.csv"
# data = 		"data/dtlz2_mod_pof_3_190.csv"
# data = 		"data/dtlz2_pof_3_190.csv"
# data = 		"data/dtlz7_pof_3_190.csv"
data =		"../../data/do2dk-k4-norm.csv"
# data =		"../../data/do2dk-norm.csv"

reset
set datafile separator ","
# splot data, knees, aug_knees
# splot data, knees
# splot data
plot data
