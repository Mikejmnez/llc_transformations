import numpy as np
import xarray as xr
from set_dims import Dims


class LLCtransformation:
    """ A class containing the transformation of LLCgrids"""

    def __init__(self, ds, varlist, transformation, centered, faces='all'):
        self._ds = ds  # xarray.DataSet
        self._varlist = varlist  # variables names to be transformed
        self._transformation = transformation  # str - type of transf
        self._centered = centered  # str - where to be centered
        self._faces = faces  # faces involved in transformation


def pos_chunks(faces, arc_faces, chunksY, chunksX):
    nrotA = [k for k in range(3)]
    nrotB = [k for k in range(3, 6)]
    nrot = nrotA + nrotB
    rotA = [k for k in range(7, 10)]
    rotB = [k for k in range(10, 13)]
    rot = rotA + rotB

    nrot_A = [k for k in faces if k in nrotA]
    nrot_B = [k for k in faces if k in nrotB]
    rot_A = [k for k in faces if k in rotA]
    rot_B = [k for k in faces if k in rotB]

    ny_nApos = len(nrot_A)
    ny_nBpos = len(nrot_B)

    ny_Apos = len(rot_A)
    ny_Bpos = len(rot_B)

    POSY = []
    POSX = []

    for k in faces:
        if k in nrot:
            if k in nrot_A:
                xk = 0
                yk = 0
                if ny_nApos == 1:
                    yk = 0
                elif ny_nApos == 2:
                    if k == nrot_A[0]:
                        yk = 0
                    else:
                        yk = 1
                elif ny_nApos == 3:
                    if k == nrotA[0]:
                        yk = 0
                    elif k == nrotA[1]:
                        yk = 1
                    elif k == nrotA[2]:
                        yk = 2
            elif k in nrot_B:
                if ny_nApos > 0:
                    xk = 1
                else:
                    xk = 0
                if ny_nBpos == 1:
                    yk = 0
                elif ny_nBpos == 2:
                    if k == nrot_B[0]:
                        yk = 0
                    else:
                        yk = 1
                elif ny_nBpos == 3:
                    if k == nrotB[0]:
                        yk = 0
                    elif k == nrotB[1]:
                        yk = 1
                    elif k == nrotB[2]:
                        yk = 2
        elif k in rot:
            if k in rot_A:
                xk = 0
                yk = 0
                if ny_Apos == 1:
                    xk = 0
                elif ny_Apos == 2:
                    if k == rot_A[0]:
                        xk = 0
                    else:
                        xk = 1
                elif ny_Apos == 3:
                    if k == rotA[0]:
                        xk = 0
                    elif k == rotA[1]:
                        xk = 1
                    elif k == rotA[2]:
                        xk = 2
            elif k in rot_B:
                if ny_Apos > 0:
                    yk = 1
                else:
                    yk = 0
                if ny_Bpos == 1:
                    xk = 0
                elif ny_Bpos == 2:
                    if k == rot_B[0]:
                        xk = 0
                    else:
                        xk = 1
                elif ny_Bpos == 3:
                    if k == rotB[0]:
                        xk = 0
                    elif k == rotB[1]:
                        xk = 1
                    elif k == rotB[2]:
                        xk = 2
        else:
            print('face index not in LLC grid')
        POSY.append(chunksY[yk])
        POSX.append(chunksX[xk])
    # This to create a new list with positions for Arctic cap slices
    POSY_arc = []
    POSX_arc = []

    aface_nrot = [k for k in arc_faces if k in nrotA + nrotB]
    aface_rot = [k for k in arc_faces if k in rotA + rotB]

    if len(aface_rot) == 0:
        if len(aface_nrot) == 0:
            print('no arctic faces')
        else:
            pos_r = chunksY[-1][-1]
            pos_l = chunksY[-1][0]
            if len(aface_nrot) == 1:
                POSX_arc.append(chunksX[0])
                POSY_arc.append([pos_r, int(pos_r + (pos_r - pos_l) / 2)])
            elif len(aface_nrot) == 2:
                for k in range(len(aface_nrot)):
                    POSX_arc.append(chunksX[k])
                    POSY_arc.append([pos_r, int(pos_r + (pos_r - pos_l) / 2)])
    else:
        if len(aface_rot) == 1:
            POSY_arc.append(chunksY[0])
        else:
            for k in range(len(aface_rot)):
                POSY_arc.append(chunksY[k])
        POSX_arc.append([0, chunksX[0][0]])
    return POSY, POSX, POSY_arc, POSX_arc


def chunk_sizes(faces, Nx, Ny, rotated=False):
    '''
    Determines the total size of array that will connect all rotated or
    non-rotated faces
    '''
    if rotated is False:
        A_ref = np.array([k for k in range(3)])
        B_ref = np.array([k for k in range(3, 6)])
    elif rotated is True:
        A_ref = np.array([k for k in range(7, 10)])
        B_ref = np.array([k for k in range(10, 13)])

    A_list = [k for k in faces if k in A_ref]
    B_list = [k for k in faces if k in B_ref]

    if len(A_list) == 0:
        if len(B_list) > 0:
            tNx = Nx[0]
            if len(B_list) == 1:
                tNy = Ny[0]
            elif len(B_list) == 2:
                if min(B_list) == B_ref[0] and max(B_list) == B_ref[-1]:
                    print('error, these faces do not connect. Not possible to create a single dataset that minimizes nans')
                else:
                    tNy = len(B_list) * Ny[0]
            else:
                tNy = len(B_list) * Ny[0]
        else:
            tNx = 0
            tNy = 0
            print('error, no data survives the cutout. Change the values')
    else:
        if len(B_list) == 0:
            tNx = Nx[0]
            if len(A_list) == 1:
                tNy = Ny[0]
            elif len(A_list) == 2:
                if min(A_list) == A_ref[0] and max(A_list) == A_ref[-1]:
                    print('error, these faces do not connect. Not possible to create a single datase that minimizes nans')
                    tNy = 0
                else:
                    tNy = len(A_list) * Ny[0]
            else:
                tNy = len(A_list) * Ny[0]
        elif len(B_list) > 0:
            tNx = 2 * Nx[0]
            if len(B_list) == len(A_list):
                if len(A_list) == 1:
                    iA = [np.where(faces[nk] == A_ref)[0][0] for nk in range(len(faces)) if faces[nk] in A_ref]
                    iB = [np.where(faces[nk] == B_ref)[0][0] for nk in range(len(faces)) if faces[nk] in B_ref] 
                    if iA == iB:
                        tNy = Ny[0]
                    else:
                        tNy = 0
                        print('Error, faces do not connect within facet')
                elif len(A_list) == 2:
                    if min(A_list) == A_ref[0] and max(A_list) == A_ref[-1]:
                        print('Error, faces do not connect within facet')
                        tNy = 0
                    else:
                        iA = [np.where(faces[nk] == A_ref)[0][0] for nk in range(len(faces)) if faces[nk] in A_ref]
                        iB = [np.where(faces[nk] == B_ref)[0][0] for nk in range(len(faces)) if faces[nk] in B_ref] 
                        if iA == iB:
                            tNy = len(A_list) * Ny_nrot[0]
                        else:
                            print('error, not all faces connect equally')
                            tNy = 0
                else:
                    tNy = len(A_list) * Ny[0]
            else:
                tNy = 0
                print('Number of faces in facet A is not equal to the number of faces in facet B.')
    return tNy, tNx


def face_connect(ds, all_faces):
    '''
    Determines the size of the final array consisting of connected faces. Does
    not consider the Arctic, since the Arctic cap is treated separatedly.
    '''
    arc_cap = 6
    Xdim = 'X'
    Ydim = 'Y'

    Nx_nrot = []
    Ny_nrot = []
    Nx_rot = []
    Ny_rot = []

    transpose = np.arange(7,13)
    nrot_faces = []
    rot_faces = []

    xpos = 0
    ypos = 0

    for k in [ii for ii in all_faces if ii not in [arc_cap]]:
        if k in transpose:
            x0, xf = 0, int(len(ds[Xdim]))
            y0, yf = 0, int(len(ds[Ydim]))
            Nx_rot.append(len(ds[Xdim][x0:xf]))
            Ny_rot.append(len(ds[Ydim][y0:yf]))
            rot_faces.append(k)
        else:
            x0, xf = 0, int(len(ds[Xdim]))
            y0, yf = 0, int(len(ds[Ydim]))
            Nx_nrot.append(len(ds[Xdim][x0:xf]))
            Ny_nrot.append(len(ds[Ydim][y0:yf]))
            nrot_faces.append(k)
    return nrot_faces, Nx_nrot, Ny_nrot, rot_faces, Nx_rot, Ny_rot


def arct_connect(ds, varName, all_faces):
    arc_cap = 6
    Nx_ac_nrot = []
    Ny_ac_nrot = []
    Nx_ac_rot = []
    Ny_ac_rot = []
    ARCT = []
    arc_faces = []
    metrics = ['dxC', 'dyC', 'dxG', 'dyG']

    if arc_cap in all_faces:
        for k in all_faces:
            if k == 2:
                fac = 1
                arc_faces.append(k)
                _varName = varName
                DIMS = [dim for dim in ds[_varName].dims if dim != 'face']
                dims = Dims(DIMS[::-1])
                dtr = list(dims)[::-1]
                dtr[-1], dtr[-2] = dtr[-2], dtr[-1]
                mask2 = xr.ones_like(ds[_varName].isel(face=arc_cap))
                # TODO: Eval where, define argument outside
                mask2 = mask2.where(np.logical_and(ds[dims.X] < ds[dims.Y],
                                                   ds[dims.X] < len(ds[dims.Y]) - ds[dims.Y]))
                x0, xf = 0, int(len(ds[dims.Y]) / 2)  # TODO: CHECK here!
                y0, yf = 0, int(len(ds[dims.X]))
                xslice = slice(x0, xf)
                yslice = slice(y0, yf)
                Nx_ac_nrot.append(0)
                Ny_ac_nrot.append(len(ds[dims.Y][y0:yf]))
                da_arg = {'face': arc_cap, dims.X: xslice, dims.Y: yslice}
                sort_arg = {'variables': dims.Y, 'ascending': False}
                mask_arg = {dims.X: xslice, dims.Y: yslice}
                if len(dims.X) + len(dims.Y) == 4:
                    if len(dims.Y) == 1 and _varName not in metrics:
                        fac = - 1
                    if 'mates' in list(ds[_varName].attrs):
                        _varName = ds[_varName].attrs['mates']
                    _DIMS = [dim for dim in ds[_varName].dims if dim != 'face']
                    dims = Dims(_DIMS[::-1])
                    dtr = list(dims)[::-1]
                    dtr[-1], dtr[-2] = dtr[-2], dtr[-1]
                    mask2 = xr.ones_like(ds[_varName].isel(face=arc_cap))
                    mask2 = mask2.where(np.logical_and(ds[dims.X] < ds[dims.Y],
                                                       ds[dims.X] < len(ds[dims.Y]) - ds[dims.Y]))
                    da_arg = {'face': arc_cap, dims.X: xslice, dims.Y: yslice}
                    sort_arg = {'variables': dims.Y, 'ascending': False}
                    mask_arg = {dims.X: xslice, dims.Y: yslice}
                arct = fac * ds[_varName].isel(**da_arg)
                arct = arct.sortby(**sort_arg)
                Mask = mask2.isel(**mask_arg)
                Mask = Mask.sortby(**sort_arg)
                arct = (arct * Mask).transpose(*dtr)
                ARCT.append(arct)

            elif k == 5:
                fac = 1
                arc_faces.append(k)
                _varName = varName
                DIMS = [dim for dim in ds[_varName].dims if dim != 'face']
                dims = Dims(DIMS[::-1])
                mask5 = xr.ones_like(ds[_varName].isel(face=arc_cap))
                mask5 = mask5.where(np.logical_and(ds[dims.X] > ds[dims.Y],
                                                   ds[dims.X] < len(ds[dims.Y]) - ds[dims.Y]))
                x0, xf = 0, int(len(ds[dims.X]))
                y0, yf = 0, int(len(ds[dims.Y]) / 2)
                xslice = slice(x0, xf)
                yslice = slice(y0, yf)
                Nx_ac_nrot.append(0)
                Ny_ac_nrot.append(len(ds[dims.X][y0:yf]))
                da_arg = {'face': arc_cap, dims.X: xslice, dims.Y: yslice}
                mask_arg = {dims.X: xslice, dims.Y: yslice}
                arct = ds[_varName].isel(**da_arg)
                Mask = mask5.isel(**mask_arg)
                arct = (arct * Mask)
                ARCT.append(arct)

            elif k == 7:
                fac = 1
                arc_faces.append(k)
                _varName = varName
                DIMS = [dim for dim in ds[_varName].dims if dim != 'face']
                dims = Dims(DIMS[::-1])
                dtr = list(dims)[::-1]
                dtr[-1], dtr[-2] = dtr[-2], dtr[-1]
                mask7 = xr.ones_like(ds[_varName].isel(face=arc_cap))
                mask7 = mask7.where(np.logical_and(ds[dims.X] > ds[dims.Y],
                                                   ds[dims.X] > len(ds[dims.Y]) - ds[dims.Y]))
                x0, xf = int(len(ds[dims.Y]) / 2), int(len(ds[dims.Y]))
                y0, yf = 0, int(len(ds[dims.X]))
                xslice = slice(x0, xf)
                yslice = slice(y0, yf)
                Nx_ac_rot.append(len(ds[dims.Y][x0:xf]))
                Ny_ac_rot.append(0)
                if len(dims.X) + len(dims.Y) == 4:
                    if len(dims.X) == 1 and _varName not in metrics:
                        fac = - 1
                    if 'mates' in list(ds[varName].attrs):
                        _varName = ds[varName].attrs['mates']
                    DIMS = [dim for dim in ds[_varName].dims if dim != 'face']
                    dims = Dims(DIMS[::-1])
                    dtr = list(dims)[::-1]
                    dtr[-1], dtr[-2] = dtr[-2], dtr[-1]
                    mask7 = xr.ones_like(ds[_varName].isel(face=arc_cap))
                    mask7 = mask7.where(np.logical_and(ds[dims.X] > ds[dims.Y],
                                                       ds[dims.X] > len(ds[dims.Y]) - ds[dims.Y]))
                da_arg = {'face': arc_cap, dims.X: xslice, dims.Y: yslice}
                mask_arg = {dims.X: xslice, dims.Y: yslice}
                sort_arg = {'variables': [dims.X], 'ascending': False}
                arct = fac * ds[_varName].isel(**da_arg)
                arct = arct.sortby(**sort_arg)
                Mask = mask7.isel(**mask_arg)
                arct = (arct * Mask).transpose(*dtr)
                ARCT.append(arct)

            elif k == 10:
                fac = 1
                _varName = varName
                DIMS = [dim for dim in ds[_varName].dims if dim != 'face']
                dims = Dims(DIMS[::-1])
                arc_faces.append(k)
                mask10 = xr.ones_like(ds[_varName].isel(face=arc_cap))
                mask10 = mask10.where(np.logical_and(ds[dims.X] < ds[dims.Y],
                                                     ds[dims.X] > len(ds[dims.Y]) - ds[dims.Y]))
                x0, xf = 0, int(len(ds[dims.X]))
                y0, yf = int(len(ds[dims.Y]) / 2), int(len(ds[dims.Y]))
                xslice = slice(x0, xf)
                yslice = slice(y0, yf)
                Nx_ac_rot.append(0)
                Ny_ac_rot.append(len(ds[dims.Y][y0:yf]))
                if len(dims.X) + len(dims.Y) == 4:
                    if _varName not in metrics:
                        fac = -1
                da_arg = {'face': arc_cap, dims.X: xslice, dims.Y: yslice}
                sort_arg = {'variables': [dims.X, dims.Y], 'ascending': False}
                mask_arg = {dims.X: xslice, dims.Y: yslice}
                arct = fac * ds[_varName].isel(**da_arg)
                arct = arct.sortby(**sort_arg)
                Mask = mask10.isel(**mask_arg)
                Mask = Mask.sortby(**sort_arg)
                arct = (arct * Mask)
                ARCT.append(arct)

    return arc_faces, Nx_ac_nrot, Ny_ac_nrot, Nx_ac_rot, Ny_ac_rot, ARCT


nrot_faces, Nx_nrot, Ny_nrot, rot_faces, Nx_rot, Ny_rot = face_connect(ds, all_faces)

if isinstance(varlist, list):
    varName=varlist[0]
elif isinstance(varlist, str):
    varName=varlist

arc_faces, Nx_ac_nrot, Ny_ac_nrot, Nx_ac_rot, Ny_ac_rot, ARCT = arct_connect(ds, varName, all_faces)

acnrot_faces = [k for k in arc_faces if k in np.array([2,5])]
acrot_faces = [k for k in arc_faces if k in np.array([7,10])]
nrot_A = [k for k in nrot_faces if k in np.arange(3)]
nrot_B = [k for k in nrot_faces if k in np.arange(3,6)]
rot_A = [k for k in rot_faces if k in np.arange(7,10)]
rot_B = [k for k in rot_faces if k in np.arange(10,13)]

tNy_nrot, tNx_nrot = chunk_sizes(nrot_faces, [Nx], [Ny])
tNx_rot, tNy_rot = chunk_sizes(rot_faces, [Nx], [Ny], rotated=True)


#Set total size of array
delNX = 0
delNY = 0
if len(ARCT)>0:
    delNX = int(Nx/2) #for rotated
    delNY = int(Ny/2) # for nonrotated
tNy_nrot = tNy_nrot + delNY # On non-rotated faces, the including the arctic
tNx_rot = tNx_rot + delNX

## Non-Rotated
Nx_nrot = np.arange(0,tNx_nrot+1,Nx)
Ny_nrot = np.arange(0,tNy_nrot+1,Ny)
chunksX_nrot = []
chunksY_nrot = []
for ii in range(len(Nx_nrot)-1):
    chunksX_nrot.append([Nx_nrot[ii],Nx_nrot[ii+1]])
for jj in range(len(Ny_nrot)-1):
    chunksY_nrot.append([Ny_nrot[jj],Ny_nrot[jj+1]])
    POSY_nrot, POSX_nrot, POSYarc_nrot, POSXarc_nrot = pos_chunks(nrot_faces, acnrot_faces,chunksY_nrot, chunksX_nrot)

## Rotated
Nx_rot = np.arange(delNX,tNx_rot+1,Nx)
Ny_rot = np.arange(0,tNy_rot+1,Ny)

chunksX_rot=[]
chunksY_rot=[]
for ii in range(len(Nx_rot)-1):
    chunksX_rot.append([Nx_rot[ii],Nx_rot[ii+1]])
for jj in range(len(Ny_rot)-1):
    chunksY_rot.append([Ny_rot[jj],Ny_rot[jj+1]])

POSY_rot, POSX_rot, POSYa_rot, POSXa_rot = pos_chunks(rot_faces, acrot_faces, chunksY_rot, chunksX_rot)

## Creates Arrays compatible in dimensions for rotated and non-rotated facets

centered_ON = 'Atlantic'
# centered_ON = 'Pacific'
X0=0
Xr0=0
if centered_ON == 'Atlantic':
    X0=tNy_rot
    
elif centered_ON == 'Pacific':
    Xr0 = tNx_nrot

print(['(X0,Xr0)',X0, Xr0])

# TODO: Crete variables here (initialized them as empty). Easier than what I am doing after, and thus avoiding creating very large np.arrays (zeros) of Order of Tbytes)

coords_nrot={'X':(('X',), np.arange(X0,X0+tNx_nrot),{'axis': 'X'}),
             'Xp1':(('Xp1',), np.arange(X0,X0+tNx_nrot),{'axis':'X'}),
             'Y':(('Y',), np.arange(tNy_nrot),{'axis': 'Y'}),
             'Yp1':(('Yp1',), np.arange(tNy_nrot), {'axis':'Y'}),
             'Z': (('Z',), np.arange(len(ds['Z'])), {'axis': 'Z'}),
             'Zp1':(('Zp1',), np.arange(len(ds['Zp1'])), {'axis':'Z'}),
             'Zl': (('Zl',), np.arange(len(ds['Zl'])), {'axis':'Z'}),
             'time': (('time',), ds['time'].data, {'axis':'T'}),   
                             }
NR_dsnew = xr.Dataset(coords=coords_nrot)
for dim in NR_dsnew.dims:
    NR_dsnew[dim].attrs = ds[dim].attrs


# ### Rotated dataset (facets)
coords_rot={'X':(('X',), np.arange(Xr0, Xr0 + tNy_rot),{'axis': 'X'}),
            'Xp1':(('Xp1',), np.arange(Xr0, Xr0 + tNy_rot), {'axis':'X'}),
            'Y':(('Y',), np.arange(tNx_rot),{'axis': 'Y'}),
            'Yp1':(('Yp1',), np.arange(tNx_rot),{'axis':'Y'}),
            'Z': (('Z',), np.arange(len(ds['Z'])), {'axis': 'Z'}),
            'Zp1':(('Zp1',), np.arange(len(ds['Zp1'])), {'axis':'Z'}), 
            'Zl': (('Zl',), np.arange(len(ds['Zl'])), {'axis':'Z'}),
            'time': (('time',), ds['time'].data, {'axis':'T'}),                
                             }
R_dsnew = xr.Dataset(coords=coords_rot)
for dim in R_dsnew.dims:
    R_dsnew[dim].attrs = ds[dim].attrs



## Transform vars: Arctic Crown only
metrics = ['dxC', 'dyC', 'dxG', 'dyG']
for varName in ds.data_vars:
    vName = varName
    print(varName)
    fac = 1
    dims = Dims([dim for dim in ds[varName].dims if dim != 'face'][::-1])
    if len(ds[varName].dims) == 1:
        R_dsnew[varName] =(dims._vars[::-1], ds[varName].data)
        NR_dsnew[varName] = (dims._vars[::-1],ds[varName].data)
        NR_dsnew[varName].attrs = od._ds[varName].attrs
        R_dsnew[varName].attrs = od._ds[varName].attrs
    else: 
        _shape = tuple([len(NR_dsnew[var]) for var in tuple(dims)[::-1]])
        NR_dsnew[varName] = (dims._vars[::-1], np.nan * np.zeros(_shape))
        R_dsnew[varName] = (dims._vars[::-1], np.nan * np.zeros(_shape))
        NR_dsnew[varName].attrs = od._ds[varName].attrs
        R_dsnew[varName].attrs = od._ds[varName].attrs
        if len(dims.X) + len(dims.Y) == 4:  # vector fields
            if 'mates' in list(ds[varName].attrs):
                vName = ds[varName].attrs['mates']
            if len(dims.X) == 1 and varName not in metrics: ## all variables except dxG, dyC, dyG
                fac = -1
        arc_faces, Nx_ac_nrot, Ny_ac_nrot, Nx_ac_rot, Ny_ac_rot, ARCT = arct_connect(ds, varName, all_faces) ## If feed U for face 
        for k in range(len(nrot_faces)):
            data = od._ds[varName].isel(face=nrot_faces[k]).values
            xslice = slice(POSX_nrot[k][0], POSX_nrot[k][1])
            yslice = slice(POSY_nrot[k][0], POSY_nrot[k][1])
            arg={dims.X:xslice, dims.Y:yslice}
            NR_dsnew[varName].isel(**arg)[:] = data
        for k in range(len(rot_faces)):
            kk = len(rot_faces)-(k+1)
            xslice = slice(POSX_nrot[k][0], POSX_nrot[k][1])
            yslice = slice(POSY_nrot[kk][0], POSY_nrot[kk][1])
            data = fac * od._ds[vName].isel(face=rot_faces[k])  ## initially U samples V
            arg={dims.Y:yslice, dims.X:xslice} # dims of original
            dtr=list(dims)[::-1]
            dtr[-1], dtr[-2] = dtr[-2], dtr[-1]
            ndims = Dims(list(data.dims)[::-1])
            sort_arg={'variables':ndims.X, 'ascending':False}
            data = data.sortby(**sort_arg)
            R_dsnew[varName].isel(**arg).transpose(*dtr)[:] = data.values # save into U
        for k in range(len(acnrot_faces)):  ###
            data=ARCT[k]
            xslice = slice(POSXarc_nrot[k][0], POSXarc_nrot[k][1])
            yslice = slice(POSYarc_nrot[k][0], POSYarc_nrot[k][1])
            data =  ARCT[k]
            arg={dims.X:xslice, dims.Y:yslice}
            NR_dsnew[varName].isel(**arg)[:] = data.values
        for k in range(len(acrot_faces)):
            tk = len(acnrot_faces) + k
            xslc = slice(POSXarc_nrot[k][0], POSXarc_nrot[k][1])
            yslc = slice(POSYarc_nrot[k][0], POSYarc_nrot[k][1])
            arg={dims.Y:yslc, dims.X:xslc}
            data = ARCT[tk] #
            R_dsnew[varName].isel(**arg)[:] = data.values


# combine them according to before
if centered_ON == 'Atlantic':
    DS=R_dsnew.combine_first(NR_dsnew)
# elif centered_ON == 'Pacific':
    DS=NR_dsnew.combine_first(R_dsnew)
DS=DS.reset_coords()


## return DS
