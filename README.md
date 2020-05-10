# llc_transformations

A class that defines several transformations to model data output from the MITgcm on the cube-sphere (LLC-grid). The set of functions transforms the data by eliminating the dimension 'face' on each variable, hence recovering all of ```xgcm```'s functionality (__e.g.__ ``grid.interp`` and ``grind.diff``).