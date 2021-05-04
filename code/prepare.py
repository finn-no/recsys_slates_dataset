#%%
import os
import logging
logging.basicConfig(format='%(asctime)s %(message)s', level='INFO')
import pickle
from torch.utils.data import Dataset, DataLoader
import torch

def mkdir(path):
    if os.path.isdir(path) == False:
        os.makedirs(path)

#%% DATALOADERS
class SequentialDataset(Dataset):
    '''
     Note: displayType has been uncommented for future easy implementation.
    '''
    def __init__(self, data, sample_uniform_action=False):

        self.data = data
        self.num_items = self.data['action'].max()+1
        self.sample_uniform_action = sample_uniform_action
        logging.info(f"Loading dataset with action size={self.data['action'].size()} and uniform candidate sampling={self.sample_uniform_action}")

    def __getitem__(self, idx):
        batch = {key: val[idx] for key, val in self.data.items()}

        if self.sample_uniform_action:
            # Sample actions uniformly:
            action = torch.randint_like(batch['action'], low=3, high=self.num_items)
            
            # Add noclick action at pos0 
            # and the actual click action at pos 1 (unless noclick):
            action[:,0] = 1
            clicked = batch['click']!=1
            action[:,1][clicked] = batch['click'][clicked]
            batch['action'] = action
            # Set click idx to 0 if noclick, and 1 otherwise:
            batch['click_idx'] = clicked.long()
            
            
            
        return batch

    def __len__(self):
        return len(self.data['click'])

#%% PREPARE DATA IN TRAINING
def load_dataloaders(data_dir,
                     batch_size=1024,
                     split_trainvalid=0.95,
                     num_workers=0,
                     override_candidate_sampler=None,
                     t_testsplit = 5,
                     sample_uniform_action=False):

    logging.info('Load data..')
    data = torch.load(f'{data_dir}/data.pt')
    dataset = SequentialDataset(data, sample_uniform_action)
    
    with open(f'{data_dir}/ind2val.pickle', 'rb') as handle:
        ind2val = pickle.load(handle)

    num_validusers = int(len(dataset) * (1-split_trainvalid))
    num_testusers = int(len(dataset) * (1-split_trainvalid))
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

    dataloaders = {
        phase: DataLoader(ds, batch_size=batch_size, shuffle=True)
        for phase, ds in subsets.items()
    }
    for key, dl in dataloaders.items():
        logging.info(
            f"In {key}: num_users: {len(dl.dataset)}, num_batches: {len(dl)}"
        )


    with open(f'{data_dir}/itemattr.pickle', 'rb') as handle:
        itemattr = pickle.load(handle)

    return ind2val, itemattr, dataloaders
