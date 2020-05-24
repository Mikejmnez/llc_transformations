import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

def llc_plots(od,varName,plotType='pcolormesh',**kwargs):
    data = od._ds[varName]
    X_name = [dim for dim in od.grid_coords['X'] if dim in data.dims][0]
    Y_name = [dim for dim in od.grid_coords['Y'] if dim in data.dims][0]
    
    # Create masking and spread Arctic Cap across faces
    mask10 = xr.ones_like(od._ds[varName].isel(face=6)) # above face 10, position = [0,1]
    mask7 = xr.ones_like(od._ds[varName].isel(face=6)) # above face 7, position = [0,0]
    mask5 = xr.ones_like(od._ds[varName].isel(face=6)) # above face 5, position = [0,3]
    mask2 = xr.ones_like(od._ds[varName].isel(face=6)) # above face 2, position = [0,2]

    mask10 = mask10.where(np.logical_and(od._ds[X_name] < od._ds[Y_name], od._ds[X_name] > len(od._ds[Y_name]) - od._ds[Y_name]))
    mask7 = mask7.where(np.logical_and(od._ds[X_name] > od._ds[Y_name], od._ds[X_name] > len(od._ds[Y_name]) - od._ds[Y_name]))
    mask5 = mask5.where(np.logical_and(od._ds[X_name] > od._ds[Y_name], od._ds[X_name] < len(od._ds[Y_name]) - od._ds[Y_name] ))
    mask2 = mask2.where(np.logical_and(od._ds[X_name] < od._ds[Y_name], od._ds[X_name] < len(od._ds[Y_name]) - od._ds[Y_name]))

    
    face_to_axis = {0: (3, 2), 1: (2, 2), 2: (1, 2),
                    3: (3, 3), 4: (2, 3), 5: (1, 3),
                    7: (1, 0), 8: (2, 0), 9: (3, 0), 
                    10: (1, 1), 11: (2, 1), 12: (3, 1), 
                    13: (0, 0), 14: (0, 1), # masked arctic cap
                    15: (0, 2), 16: (0,3), # masked arctic cap
                    }
    transpose = [k for k in range(7,13)]
    arctic_cap = [k for k in range(13,17)]
    gridspec_kw = dict(left=0, bottom=0, right=1, top=1, wspace=0.0005, hspace=0.00005)
    xincrease=True
    yincrease=True
    fig, axes = plt.subplots(nrows=4, ncols=4, gridspec_kw=gridspec_kw,figsize=(12,8))
    Xdim = X_name
    Ydim = Y_name
    for FACE, (j,i) in face_to_axis.items():
        ax = axes[j,i]
        if FACE in arctic_cap:
            if FACE == 13: # position [0,0]
                Xdim = Y_name
                Ydim = X_name
                xincrease = True
                yincrease = False
                x0,xf = int(len(od._ds[Xdim])/2),int(len(od._ds[Xdim]))
                y0,yf = 0,int(len(od._ds[Ydim]))
                xslice=slice(x0,xf)
                yslice=slice(y0,yf)
                da_face = (od._ds[varName].isel(face=6,X=xslice,Y=yslice)[::2,::2]) * mask7.isel(X=xslice,Y=yslice)[::2,::2]
            elif FACE == 14: # position [0,1]
                xincrease = False
                yincrease = False
                Xdim = X_name
                Ydim = Y_name
                x0,xf = 0,int(len(od._ds[Xdim]))
                y0,yf = int(len(od._ds[Ydim])/2),int(len(od._ds[Ydim]))
                xslice=slice(x0,xf)
                yslice=slice(y0,yf)
                da_face = (data.isel(face=6,X=xslice, Y=yslice)[::2,::2]) *  (mask10.isel(X=xslice,Y=yslice)[::2,::2])

            elif FACE == 15: # position [0,2]
                xincrease = False
                yincrease = True
                Xdim = Y_name
                Ydim = X_name
                x0,xf = 0, int(len(od._ds[Xdim])/2)
                y0,yf = 0, int(len(od._ds[Ydim]))
                xslice=slice(x0,xf)
                yslice=slice(y0,yf)
                da_face = (data.isel(face=6,X=xslice,Y=yslice)[::2,::2]) * (mask2.isel(X=xslice,Y=yslice)[::2,::2])
                
            elif FACE == 16:
                xincrease = True
                yincrease = True
                Xdim = X_name
                Ydim = Y_name
                x0,xf = 0, int(len(od._ds[Xdim]))
                y0,yf = 0, int(len(od._ds[Ydim])/2)
                xslice=slice(x0,xf)
                yslice=slice(y0,yf)
                da_face = (od._ds[varName].isel(face=6,X=xslice,Y=yslice)[::2,::2]) * mask5.isel(X=xslice,Y=yslice)[::2,::2]

            args = {'ax': ax, 'x': Xdim, 'y': Ydim, 'add_colorbar': False,
                    'xincrease': xincrease, 'yincrease': yincrease,
                    **kwargs}

#         elif FACE not in data['oface'].values:
#             ax.axis('on')
#         else:
#             fc = np.where(FACE==data['oface'].values)[0][0]
#             da_face = data.isel(face=fc)[::2,::2]
#             if FACE == 6:
#                 xincrease=False
#                 yincrease = True
#                 Xdim = Y_name
#                 Ydim = X_name
        else:
            da_face = data.isel(face=FACE)[::2,::2]
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