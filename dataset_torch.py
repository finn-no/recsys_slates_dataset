#%% Imports
import torch
import datahelper
# datahelper.download_data_files()

#%%
import os
import logging
logging.basicConfig(format='%(asctime)s %(message)s', level='INFO')
import pickle
from torch.utils.data import Dataset, DataLoader
import torch
import json
def mkdir(path):
    if os.path.isdir(path) == False:
        os.makedirs(path)
import numpy as np
#%% DATALOADERS
class SequentialDataset(Dataset):
    '''
     Note: displayType has been uncommented for future easy implementation.
    '''
    def __init__(self, data, sample_uniform_slate=False):

        self.data = data
        self.num_items = self.data['slate'].max()+1
        self.sample_uniform_slate = sample_uniform_slate
        logging.info("Loading dataset with slate size={} and uniform candidate sampling={}".format(self.data['slate'].size(), self.sample_uniform_slate))

    def __getitem__(self, idx):
        batch = {key: val[idx] for key, val in self.data.items()}

        if self.sample_uniform_slate:
            # Sample actions uniformly:
            action = torch.randint_like(batch['slate'], low=3, high=self.num_items)
            
            # Add noclick action at pos0 
            # and the actual click action at pos 1 (unless noclick):
            action[:,0] = 1
            clicked = batch['click']!=1
            action[:,1][clicked] = batch['click'][clicked]
            batch['slate'] = action
            # Set click idx to 0 if noclick, and 1 otherwise:
            batch['click_idx'] = clicked.long()
            
        return batch

    def __len__(self):
        return len(self.data['click'])

#%% PREPARE DATA IN TRAINING
def load_dataloaders(data_dir,
                     batch_size=1024,
                     split_trainvalid=0.90,
                     t_testsplit = 5,
                     num_workers = 0,
                     sample_uniform_slate=False):
    
    logging.info("Download data if not in data folder..")
    datahelper.download_data_files(data_dir=data_dir)

    logging.info('Load data..')
    with np.load("{}/data.npz".format(data_dir)) as data_np:
        data = {key: torch.tensor(val) for key, val in data_np.items()}
    dataset = SequentialDataset(data, sample_uniform_slate)
    
    with open('{}/ind2val.json'.format(data_dir), 'rb') as handle:
        # Use string2int object_hook found here: https://stackoverflow.com/a/54112705
        ind2val = json.load(
            handle, 
            object_hook=lambda d: {int(k) if k.lstrip('-').isdigit() else k: v for k, v in d.items()}
            )

    # Split dataset into train, validation and test:
    num_validusers = int(len(dataset) * (1-split_trainvalid)/2)
    num_testusers = int(len(dataset) * (1-split_trainvalid)/2)
    torch.manual_seed(0)
    num_users = len(dataset)
    perm_user = torch.randperm(num_users)
    valid_user_idx = perm_user[:num_validusers]
    test_user_idx  = perm_user[num_validusers:(num_validusers+num_testusers)]
    train_user_idx = perm_user[(num_validusers+num_testusers):]
    # Mask type: 1: train, 2: valid, 3: test
    dataset.data['mask_type'] = torch.ones_like(dataset.data['click'])
    dataset.data['mask_type'][valid_user_idx, t_testsplit:] = 2
    dataset.data['mask_type'][test_user_idx, t_testsplit:] = 3

    subsets = {
        'train': dataset, 
        'valid': torch.utils.data.Subset(dataset, valid_user_idx),
        'test': torch.utils.data.Subset(dataset, test_user_idx)
        }

    # Build dataloaders for each data subset:
    dataloaders = {
        phase: DataLoader(ds, batch_size=batch_size, shuffle=(phase=="train"), num_workers=num_workers)
        for phase, ds in subsets.items()
    }
    for key, dl in dataloaders.items():
        logging.info(
            "In {}: num_users: {}, num_batches: {}".format(key, len(dl.dataset), len(dl))
        )
    
    # Load item attributes:
    with np.load('{}/itemattr.npz'.format(data_dir), mmap_mode=None) as itemattr_file:
        itemattr = {key : val for key, val in itemattr_file.items()}

    return ind2val, itemattr, dataloaders
