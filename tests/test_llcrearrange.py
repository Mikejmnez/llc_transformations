# tests for llc_rearrange.py
import pytest
from conftest import od
import sys
sys.path.append('/Users/Mikejmnez/llc_transformations/llc_rearrange/')
from LLC_rearrange import LLCtransformation as LLC
from LLC_rearrange import make_chunks, make_array, drop_size, pos_chunks, chunk_sizes, face_connect, arct_connect
from LLC_rearrange import Dims


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
