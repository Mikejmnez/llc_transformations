import numpy as _np
import xarray as _xr
from LLC_rearrange import Dims
import matplotlib.pyplot as _plt


def llc_plots(ds, varName, plotType='pcolormesh', **kwargs):
    '''
    function that plots variables within a dataset defined on an LLC grid
    '''

    X_name = [dim for dim in od.grid_coords['X'] if dim in data.dims][0]
    Y_name = [dim for dim in od.grid_coords['Y'] if dim in data.dims][0]

    # Create masking and spread Arctic Cap across faces
    # mask10: above face 10, position = [0,1]
    mask10 = _xr.ones_like(ds[varName].isel(face=6))
    # mask7: above face 7, position = [0,0]
    mask7 = xr.ones_like(od._ds[varName].isel(face=6))
    # mask5: above face 5, position = [0,3]
    mask5 = xr.ones_like(od._ds[varName].isel(face=6))
    # mask2: above face 2, position = [0,2]
    mask2 = xr.ones_like(od._ds[varName].isel(face=6))

    logical_and = _np.logical_and(ds[X_name] < ds[Y_name],
                                  ds[X_name] > len(ds[Y_name]) - ds[Y_name])
    mask10 = mask10.where(logical)

    logical_and = _np.logical_and(ds[X_name] > ds[Y_name],
                                  ds[X_name] > len(ds[Y_name]) - ds[Y_name])
    mask7 = mask7.where(logical_and)

    logical_and = _np.logical_and(ds[X_name] > ds[Y_name],
                                  ds[X_name] < len(ds[Y_name]) - ds[Y_name])
    mask5 = mask5.where(logical_and)

    logical_and = _np.logical_and(ds[X_name] < ds[Y_name],
                                  ds[X_name] < len(ds[Y_name]) - ds[Y_name])
    mask2 = mask2.where(logical_and)

    face_to_axis = {0: (3, 2), 1: (2, 2), 2: (1, 2),
                    3: (3, 3), 4: (2, 3), 5: (1, 3),
                    7: (1, 0), 8: (2, 0), 9: (3, 0),
                    10: (1, 1), 11: (2, 1), 12: (3, 1),
                    13: (0, 0), 14: (0, 1),  # masked arctic cap
                    15: (0, 2), 16: (0, 3),  # masked arctic cap
                    }

    transpose = [k for k in range(7, 13)]
    arctic_cap = [k for k in range(13, 17)]
    gridspec_kw = dict(left=0, bottom=0, right=1, top=1,
                       wspace=0.0005, hspace=0.00005)
    xincrease = True
    yincrease = True
    fig, axes = plt.subplots(nrows=4, ncols=4,
                             gridspec_kw=gridspec_kw, figsize=(12, 8))
    Xdim = X_name
    Ydim = Y_name
    for FACE, (j, i) in face_to_axis.items():
        ax = axes[j, i]
        if FACE in arctic_cap:
            if FACE == 13:  # position [0,0]
                Xdim = Y_name
                Ydim = X_name
                xincrease = True
                yincrease = False
                x0, xf = int(len(ds[Xdim]) / 2), int(len(ds[Xdim]))
                y0, yf = 0, int(len(ds[Ydim]))
                xslice = slice(x0, xf, 2)
                yslice = slice(y0, yf, 2)
                da_face = ds[varName].isel(face=6, X=xslice, Y=yslice)
                da_face = da_face * mask7.isel(X=xslice, Y=yslice)
            elif FACE == 14:  # position [0,1]
                xincrease = False
                yincrease = False
                Xdim = X_name
                Ydim = Y_name
                x0, xf = 0, int(len(ds[Xdim]))
                y0, yf = int(len(ds[Ydim]) / 2), int(len(ds[Ydim]))
                xslice = slice(x0, xf, 2)
                yslice = slice(y0, yf, 2)
                da_face = (data.isel(face=6, X=xslice, Y=yslice))
                da_face = da_face * (mask10.isel(X=xslice, Y=yslice))
            elif FACE == 15:  # position [0,2]
                xincrease = False
                yincrease = True
                Xdim = Y_name
                Ydim = X_name
                x0, xf = 0, int(len(ds[Xdim]) / 2)
                y0, yf = 0, int(len(ds[Ydim]))
                xslice = slice(x0, xf, 2)
                yslice = slice(y0, yf, 2)
                da_face = data.isel(face=6, X=xslice, Y=yslice)
                da_face = da_face * mask2.isel(X=xslice, Y=yslice)
            elif FACE == 16:
                xincrease = True
                yincrease = True
                Xdim = X_name
                Ydim = Y_name
                x0, xf = 0, int(len(ds[Xdim]))
                y0, yf = 0, int(len(ds[Ydim]) / 2)
                xslice = slice(x0, xf, 2)
                yslice = slice(y0, yf, 2)
                da_face = ds[varName].isel(face=6, X=xslice, Y=yslice)
                da_face = da_face * mask5.isel(X=xslice, Y=yslice)
            args = {'ax': ax, 'x': Xdim, 'y': Ydim, 'add_colorbar': False,
                    'xincrease': xincrease, 'yincrease': yincrease, **kwargs}
        else:
            da_face = data.isel(face=FACE)[::2, ::2]
            if FACE in transpose:
                Xdim = Y_name
                Ydim = X_name
                yincrease = False
                xincrease = True
            else:
                Xdim = X_name
                Ydim = Y_name
                yincrease = True
                xincrease = True
            args = {'ax': ax, 'x': Xdim, 'y': Ydim, 'add_colorbar': False,
                    'xincrease': xincrease, 'yincrease': yincrease,
                    **kwargs}
        plotfunc = eval('xr.plot.' + plotType)
        plotfunc(da_face, **args)
        ax.axis('off')
        ax.set_title('')
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        plt.xlabel('')
        plt.ylabel('')
