# llc_transformations

A class that defines several transformations to model data output from the MITgcm on the cube-sphere (LLC-grid). The set of functions transforms the data by eliminating the dimension 'face' on each variable, hence recovering all of ```xgcm```'s functionality (__e.g.__ ``grid.interp`` and ``grind.diff``). 

In addition to having get rid of faces as a dimension, the transformation incorporates the Arctic Cap in two different ways: Splitting the Arctic into four triangular regions that share similar orientation with faces the Arctic interacts (exchanges data) with, and (to do list) another transformation in which the data is centered at the Arctic, reflecting two different ways to work with data output.
