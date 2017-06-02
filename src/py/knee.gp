#!/usr/local/bin/gnuplot -persist

# # data = "knee3d-norm.csv"
# data = "knee4d-norm.csv"
# reset
# set datafile separator ","
# # set dgrid3d 10,10,1
# # set style data lines
# # set contour base
# splot data u 2:3:4

# data = "test.csv"
# reset
# set datafile separator ","
# # splot data

# # draw the parametric surface function with one dent
# # set hidden3d
# # set isosamples 30
# set parametric
# set trange [0:1]
# set urange [0:1]
# set vrange [0:1]
# # parametric functions
# K = 1.0
# # r(x,y) = 5.0 + (cos(2.0 * K * pi * x) + cos(2.0 * K * pi * y))
# r(u) = 5.0 + (10.0 * (u - 0.5)**2) + ((2.0/K) * cos(2.0 * K * pi * u))
# rr(u,v) = (r(u) + r(v))/2.0
# fx(u,v) = rr(u,v) * cos(u * (pi/2.0)) * cos(v * (pi/2.0))
# fy(u,v) = rr(u,v) * cos(u * (pi/2.0)) * sin(v * (pi/2.0))
# fz(u,v) = rr(u,v) * sin(u * (pi/2.0))
# # splot fx(u,v), fy(u,v), fz(u,v), data
# splot fx(u,v), fy(u,v), fz(u,v)
# # set term png enhanced
# # set output "f(x,y,z)_2.png"
# # replot

# draw parametric curve with one dent
# set parametric
# set trange [0:1]
# set urange [0:1]
# set vrange [0:1]
# # set xrange [0:1]
# # set yrange [0:1]
# K = 1.0
# # r(t) = (2.0 + cos(2.0 * K * pi * t))/2.0
# r(t) = 5.0 + (10.0 * (t - 0.5)**2) + ((2.0/K) * cos(2.0 * K * pi * t))
# fx(t) = r(t) * cos(t * (pi/2.0))
# fy(t) = r(t) * sin(t * (pi/2.0))
# plot t, r(t)
# # plot fx(t), fy(t)

M = 5
plot [0:1] M - x * (1 + sin(3.0 * pi * x))
