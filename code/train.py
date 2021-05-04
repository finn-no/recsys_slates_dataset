#%%
import torch
import utils
import pyro
import logging
import models
import numpy as np
logging.basicConfig(format='%(asctime)s %(message)s', level='INFO')
import dataset
import pyrotrainer as pyrotrainer
import os
import random
import time
#%%

def load_data(**kwargs):
    """
    - **kwargs will add/overwrite any set parameters in param.
    """
    param = utils.load_param(**kwargs)
        
    if param['device'] == "cuda":
        # Find optimal device and use it
        sleeptime = random.randint(0,3)
        time.sleep(sleeptime)
        if os.environ.get('interactive_v2') =="true": #only run this in interactive as production pods doesnt support it
            param['device'] = utils.get_best_cuda()
            torch.cuda.set_device(param['device'])
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    ind2val, itemattr, dataloaders = dataset.load_dataloaders(
            data_dir=param['data_dir'],
            batch_size=param['batch_size'],
            split_trainvalid=param['split_trainvalid'],
            t_testsplit = param['t_testsplit'],
            sample_uniform_action = param.get('sample_uniform_action', False)
            )
        
    noclick_rate = (dataloaders['train'].dataset.data['click']==1).float().sum()/(dataloaders['train'].dataset.data['click']>0).float().sum()
    logging.info(f"Loaded dataset has {noclick_rate*100:.1f}% of noClick.")

    param['num_items'] = len(ind2val['itemId'])
    param['num_groups'] = len(np.unique(itemattr['category']))
    param['num_users'], param['maxlen_time'], _ = dataloaders['train'].dataset.data['action'].size()
    param['num_users'] = dataloaders['train'].dataset.data['userId'].max()+1
    return param, ind2val, itemattr, dataloaders



def initialize_model(param, ind2val, itemattr, dataloaders):
    pyro.clear_param_store()
    pyro.validation_enabled(False)
    torch.manual_seed(param['train_seed'])
    
    dummybatch = next(iter(dataloaders['train']))
    dummybatch['phase_mask'] = (dummybatch['mask_type']==1).float()
    dummybatch = {key: val.long().to(param.get("device")) for key, val in dummybatch.items()}
    if param.get('remove_item_group'):
        itemattr['category'] = itemattr['category']*0
    model = models.PyroRecommender(**param, item_group=torch.tensor(itemattr['category']).long())
    guide = models.MeanFieldGuide(model=model, batch=dummybatch, **param)

    return model, guide
    
def train(param, ind2val, itemattr, dataloaders, model, guide, run_training_loop=True):
    # Common callbacks:
    optim = pyrotrainer.SviStep(model=model, guide=guide, **param)

    # Callbacks during training:
    step_callbacks = [optim, pyrotrainer.CalcBatchStats(**param)]

    phase_end_callbacks = [
        pyrotrainer.report_phase_end, 
        pyrotrainer.ReportPyroParameters(), 
        pyrotrainer.AlternateTrainingScheduleCheckpoint(model=model,guide=guide, svi_step_callback=optim,**param),
        ]

    after_training_callbacks = [pyrotrainer.calc_hitrate]
    after_training_callbacks.append(pyrotrainer.VisualizeEmbeddings())
    after_training_callbacks.append(pyrotrainer.ReportHparam(param))

    #%%
    trainer = pyrotrainer.PyroTrainer(
        model, 
        guide, 
        dataloaders, 
        before_training_callbacks = [pyrotrainer.checksum_data],
        after_training_callbacks = after_training_callbacks,
        step_callbacks = step_callbacks, 
        phase_end_callbacks = phase_end_callbacks,
        max_epoch=param['max_epochs'],
        **param)

    guide.trainer = trainer
    
    if run_training_loop:
        trainer.fit()

    return param, ind2val, trainer

def main(run_training_loop=True, **kwargs):
    param, ind2val, itemattr, dataloaders = load_data(**kwargs)
    model, guide = initialize_model(param,ind2val, itemattr, dataloaders)
    param, ind2val, trainer = train(param, ind2val, itemattr, dataloaders, model, guide)
    return param, ind2val, trainer

if __name__ == "__main__":
    kwargs = {'run_training_loop' : True}
    param, ind2val, trainer = main(**kwargs)
    
# %%
