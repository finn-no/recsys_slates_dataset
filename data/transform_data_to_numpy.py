### TRANSFORM DATA FILES FROM PYTORCH ARRAY TO NUMPY ARRAYS
# The original dataset was only available as a pytorch dataset. 
# This is unfortunate as it is less accessible for non-pytorch users.
# Further, it was also saved in pickle format, which is vulnerable to versioning.
# Lastly, some of the data names are fairly internal, change these to more understandable names.

#%% Imports
import torch
import numpy as np
import pickle


# %% Transform interaction data
# We rename the displayed items from "action" to "slate". 
# Otherwise this is just a transformation from pytorch to numpy arrays.
data_pt = torch.load("data.pt")
# Transform some of the arrays directly to numpy arrays:
transform_directly = ['userId','click','click_idx']
data_np = {key : data_pt[key].numpy() for key in transform_directly}

# Transform the displayed items with name changes of the fields:
data_np['slate_lengths'] = data_pt['lengths'].numpy()
data_np['slate'] = data_pt['action'].numpy()
data_np['slate_type'] = data_pt['displayType'].numpy()

# Save the interaction data with compresed numpy directly:
np.savez_compressed('data', **data_np)

# %% Transform the index file (ind2val):
# userId and itemId transforms are scrambled and is not useful for any purpose. 
# Remove these to reduce data size.
# Also we have renamed "displayType" to "slate_type" in data, so do same here.

ind2val_old = pickle.load(open("ind2val.pickle", "rb"))

ind2val_new = {
    'category' : ind2val_old['category'],
    'slate_type' : ind2val_old['displayType']
}
import json
with open('ind2val.json', 'w') as json_file:
    json.dump(ind2val_new, json_file)

#%% Transform item attributes (itemattr.pickle)
# Save only the category vector, and save in npz format.
itemattr_old = pickle.load(open("itemattr.pickle","rb"))
itemattr_new = {'category' : itemattr_old['category']}
np.savez_compressed('itemattr', **itemattr_new)
