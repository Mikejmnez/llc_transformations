import numpy as _np
import xarray as _xr
from LLC_rearrange import Dims, arct_connect
import matplotlib.pyplot as _plt


def llc_plots(ds, varName, plotType='pcolormesh', **kwargs):
    '''
    function that plots variables within a dataset defined on an LLC grid
    '''
    DIMS = [dim for dim in ds[varName].dims if dim != 'face'][::-1]
    dims = Dims(DIMS)
    xslice = slice(0, -1, 10)
    yslice = slice(0, -1, 10)

    A, B, C, D, E, ARCT = arct_connect(ds, varName, faces=np.arange(13))
    # ARCT is a list with 4

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
    Xdim = dims.X
    Ydim = dims.Y
    for FACE, (j, i) in face_to_axis.items():
        ax = axes[j, i]
        if FACE in arctic_cap:
            if FACE == 13:  # position [0,0]
                Xdim = dims.Y
                Ydim = dims.X
                xincrease = True
                yincrease = False
                da_face = ARCT[2]  # exchanges w face7
            elif FACE == 14:  # position [0,1]
                xincrease = False
                yincrease = False
                Xdim = dims.X
                Ydim = dims.Y
                da_face = ARCT[3]  # exchanges with face 10
            elif FACE == 15:  # position [0,2]
                xincrease = False
                yincrease = True
                Xdim = Y_name
                Ydim = X_name
                da_face = ARCT[0]  # exchanges with face 2
            elif FACE == 16:  # position [0,3] / exchanges with face 5
                xincrease = True
                yincrease = True
                Xdim = dims.X
                Ydim = dims.Y
                da_face = ARCT[1]
            args = {'ax': ax, 'x': Xdim, 'y': Ydim, 'add_colorbar': False,
                    'xincrease': xincrease, 'yincrease': yincrease, **kwargs}
        else:
            da_face = data.isel(face=FACE)
            if FACE in transpose:
                Xdim = dims.Y
                Ydim = dims.X
                yincrease = False
                xincrease = True
            else:
                Xdim = dims.X
                Ydim = dims.Y
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
