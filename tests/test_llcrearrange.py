# tests for llc_rearrange.py
import oceanspy as ospy
import sys
sys.path.append('/Users/Mikejmnez/llc_transformations/llc_rearrange/')
from LLC_rearrange import LLCtransformation as LLC
from LLC_rearrange import make_chunks, make_array, drop_size, pos_chunks, chunk_sizes, face_connect, arct_connect
from LLC_rearrange import Dims

# load sample dataset
url = '/Users/Mikejmnez/Desktop/JHU/Poseidon/Local_LLC90.yaml'
od = ospy.open_oceandataset.from_catalog('LLClocal', url)
od._ds = od._ds.drop({'k', 'k_p1', 'k_u', 'k_l'})


def test_original_dims():
    """ test original dimensions
    """
    temp = od._ds['T']
    dims = Dims([dim for dim in temp.dims][::-1])
    assert dims.X == 'X'
    assert dims.Y == 'Y'
    assert dims.Z == 'face'
    assert dims.T == 'Z'