import oceanspy as ospy
# load sample dataset
url = '/Users/Mikejmnez/Desktop/JHU/Poseidon/Local_LLC90.yaml'
od = ospy.open_oceandataset.from_catalog('LLClocal', url)
od._ds = od._ds.drop({'k', 'k_p1', 'k_u', 'k_l'})
