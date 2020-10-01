import xarray as xr
import oceanspy as ospy
from oceanspy.open_oceandataset import _find_entries
# load sample dataset

Datadir = 'data/'

vars_tiled = ['THETA', 'UVEL', 'VVEL']
ds = []
for var in vars_tiled:
    ds1 = xr.open_dataset(Datadir + var + '_2017_01.nc')
    ds.append(ds1)
ds = xr.merge(ds).drop(['XC', 'YC'])
ds = ds.drop_dims('nv').reset_coords()
ds_grid = xr.open_dataset('data/GRID/ECCO-GRID.nc').reset_coords()
ds = xr.merge([ds, ds_grid])

od = ospy.OceanDataset(ds)

url = '/Users/Mikejmnez/Desktop/JHU/Poseidon/Local_LLC90.yaml'

metadata = {}
cat, entries, url, intake_switch = _find_entries('LLClocal', url)
entry = entries[-1]
mtdt = cat[entry].metadata
rename = mtdt.pop('rename', None)
ren = {'THETA':'T', 'UVEL':'U', 'VVEL':'V'}
rename = {**rename, **ren}
od._ds = od._ds.rename(rename)
metadata = {**metadata, **mtdt}
swap_dims = mtdt.pop("swap_dims", None)
if swap_dims is not None:
    od._ds = od._ds.swap_dims(swap_dims)
for var in ['aliases', 'parameters', 'name', 'description', 'projection']:
    val = metadata.pop(var, None)
    if val is not None:
        od = eval('od.set_{}(val)'.format(var))
grid_coords = metadata.pop('grid_coords', None)
if grid_coords is not None:
    od = od.set_grid_coords(**grid_coords)
face_connections = metadata.pop('face_connections', None)
if face_connections is not None:
    od = od.set_face_connections(**face_connections)

od._ds = od._ds.drop({'k', 'k_p1', 'k_u', 'k_l'})
# print(od._ds.data_vars)
