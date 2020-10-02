# tests for llc_rearrange.py
import pytest
import numpy as _np
from conftest import od
import sys
sys.path.append('/Users/Mikejmnez/llc_transformations/llc_rearrange/')
from LLC_rearrange import LLCtransformation as LLC
from LLC_rearrange import make_chunks, make_array, drop_size, pos_chunks, chunk_sizes, face_connect, arct_connect
from LLC_rearrange import Dims


Nx = od._ds.dims['X']
Ny = od._ds.dims['Y']


@pytest.mark.parametrize(
    "od, var, expected", [
        (od, 'T', ('X', 'Y', 'face', 'Z', 'time')),
        (od, 'U', ('Xp1', 'Y', 'face', 'Z', 'time')),
        (od, 'V', ('X', 'Yp1', 'face', 'Z', 'time')),
    ]
)
def test_original_dims(od, var, expected):
    """ test original dimensions
    """
    dims = Dims([dim for dim in od._ds[var].dims][::-1])
    assert dims == expected


faces = [k for k in range(13)]
nrot_expected = [0, 1, 2, 3, 4, 5]
rot_expected = [7, 8, 9, 10, 11, 12]


@pytest.mark.parametrize(
    "od, faces, nrot_expected, rot_expected", [
        (od, faces, nrot_expected, rot_expected),
        (od, faces[3:6], nrot_expected[3:6], []),
        (od, faces[8:11], [], rot_expected[1:4])
    ]
)
def test_face_connect(od, faces, nrot_expected, rot_expected):
    ds = od._ds
    nrot_faces, a, b, rot_faces, *nn = face_connect(ds, faces)
    assert nrot_faces == nrot_expected
    assert rot_faces == rot_expected


expected = [2, 5, 7, 10]  # faces that connect with arctic cap face=6
acshape = (Nx // 2, Ny)


@pytest.mark.parametrize(
    "od, faces, expected, acshape", [
        (od, faces, expected, acshape),
        (od, faces[:2], [], []),
        (od, faces[:6], [], []),
        (od, [0, 1, 2, 6], expected[:1], acshape),
        (od, faces[:7], expected[:2], acshape),
        (od, faces[6:], expected[2:], acshape)
    ]
)
def test_arc_connect(od, faces, expected, acshape):
    ds = od._ds
    arc_faces, *a, ARCT = arct_connect(ds, 'XG', faces)
    assert arc_faces == expected
    assert len(ARCT) == len(expected)
    if len(ARCT) > 0:
        assert ARCT[0].shape == acshape  # arctic crown


@pytest.mark.parametrize(
    "faces, Nx, Ny, rot, exp_tNX, exp_tNY", [
        (faces[:6], Nx, Ny, False, 180, 270),
        (faces[6:], Nx, Ny, True, 180, 270),
        (faces[:3], Nx, Ny, False, 90, 270),
        (faces[3:6], Nx, Ny, False, 90, 270),
        (faces[7:10], Nx, Ny, True, 90, 270),
        (faces[10:], Nx, Ny, True, 90, 270),
        ([0, 2], Nx, Ny, False, None, None),
        ([1, 3], Nx, Ny, False, None, None),
        ([0, 4], Nx, Ny, False, None, None),
        ([0, 5], Nx, Ny, False, None, None),
        ([1, 3], Nx, Ny, False, None, None),
        ([1, 5], Nx, Ny, False, None, None),
        ([2, 3], Nx, Ny, False, None, None),
        ([2, 4], Nx, Ny, False, None, None),
        ([0, 1, 4, 5], Nx, Ny, False, None, None),
        ([1, 2, 3, 4], Nx, Ny, False, None, None),
        ([0, 4, 5], Nx, Ny, False, None, None),
        ([7, 10], Nx, Ny, True, 180, 90),
        ([7, 11], Nx, Ny, True, None, None),
        ([7, 12], Nx, Ny, True, None, None),
        ([8, 10], Nx, Ny, True, None, None),
        ([8, 12], Nx, Ny, True, None, None),
        ([9, 10], Nx, Ny, True, None, None),
        ([9, 11], Nx, Ny, True, None, None),
        ([7, 8, 11, 12], Nx, Ny, True, None, None),
        ([8, 9, 10, 11], Nx, Ny, True, None, None)
    ]
)
def test_chunk_sizes(faces, Nx, Ny, rot, exp_tNX, exp_tNY):
    if _is_connect(faces, rotated=rot):
        tNy, tNx = chunk_sizes(faces, [Nx], [Ny], rotated=rot)
        assert tNy == exp_tNY
        assert tNx == exp_tNX
    else:
        with pytest.raises(ValueError):
            tNy, tNx = chunk_sizes(faces, [Nx], [Ny], rotated=rot)
            assert tNy == exp_tNY
            assert tNx == exp_tNX


def _is_connect(faces, rotated=False):
    """ do faces in a facet connect? Not applicable to arc cap, and only
    applicable to either rotated or not rotated facets"""
    if rotated is False:
        A_fac = _np.array([0, 1, 2])
        B_fac = _np.array([3, 4, 5])
    elif rotated is True:
        A_fac = _np.array([7, 8, 9])
        B_fac = _np.array([10, 11, 12])
    A_list = [k for k in faces if k in A_fac]
    B_list = [k for k in faces if k in B_fac]
    cont = 1
    if len(A_list) == 0:
        if len(B_list) > 1:
            if len(B_list) == 2:
                if abs(B_list[1] - B_list[0]) > 1:
                    cont = 0
    else:
        if len(B_list) == 0:
            if len(A_list) > 1:
                if len(A_list) == 2:
                    if abs(A_list[1] - A_list[0]) > 1:
                        cont = 0
        else:
            if len(B_list) == len(A_list):
                if len(A_list) == 1:
                    iA = [_np.where(faces[k] == A_fac)[0][0]
                          for k in range(len(faces))
                          if faces[k] in A_fac]
                    iB = [_np.where(faces[k] == B_fac)[0][0]
                          for k in range(len(faces))
                          if faces[k] in B_fac]
                    if iA != iB:
                        cont = 0
                if len(A_list) == 2:
                    if abs(A_list[1] - A_list[0]) > 1:
                        cont = 0
                    else:
                        iA = [_np.where(faces[k] == A_fac)[0][0]
                              for k in range(len(faces))
                              if faces[k] in A_fac]
                        iB = [_np.where(faces[k] == B_fac)[0][0]
                              for k in range(len(faces))
                              if faces[k] in B_fac]
                        if iA != iB:
                            cont = 0
            else:
                cont = 0
    return cont


