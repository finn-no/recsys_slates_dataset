#%%
import DATAHelper
import os

import sys
import logging
logging.basicConfig(format='%(asctime)s %(message)s', level='INFO')
import PYTORCHHelper
import FINNHelper
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, IntegerType, StringType, StructType, StructField, FloatType
from pyspark.sql import Window
import datetime
import numpy as np
import pandas as pd
import pickle
import datetime
from torch.utils.data import Dataset, DataLoader
import torch
import pyro
from DATAHelper import pad_sequences
import bayesrec.utils as utils
import os

def mkdir(path):
    if os.path.isdir(path) == False:
        os.makedirs(path)

#%%
'''Build ind2val from a sequences dataframe. Needs to include the action column!'''
# Build ind2val:


def load_postcodes_to_spark(sqlContext):
    '''
    ASSUMES file postkoder.csv in base dir!
    File is received from PEM and is extracted from finn data
    However, we will only need to postnr -> fylke columns
    '''
    this_dir, this_filename = os.path.split(__file__)
    postkode_path = f"{this_dir}/postkoder.csv"
    postkode = pd.read_table(postkode_path, sep=',')[['Postnr',"Byområde2",'Kommune', 'Fylke']]
    postkode.columns = ['post_code','city_area','municipality', 'region']

    def stringify_postcode(x):
        x = str(x)
        return '0' * (4 - len(x)) + x

    postkode['post_code'] = postkode.post_code.map(stringify_postcode)
    postkode['city_area'] = postkode.city_area.map(lambda x: str(x))
    postkode['municipality'] = postkode.municipality.map(lambda x: str(x))

    postcode_spark = sqlContext.createDataFrame(postkode)
    return postcode_spark


def add_catvar_if_above_threshold(df, catvar, th=100):
    df = df.withColumn('proposed_category',
                       F.concat_ws(',', 'category', catvar))

    accepted_categories = (
        df
        .groupby('proposed_category')
        .count()
        .filter(F.col('count') > th)
        .withColumn('keep', F.lit(True))
        .drop('count')
        )

    df = (
        df
        .join(accepted_categories, on='proposed_category', how='left')
        .withColumn('category',F.when(
                F.col('keep') == 'true', F.col('proposed_category'))
                .otherwise(
                    F.col('category'))
                    )
        .drop('keep').drop('proposed_category')
        )
    return df


def build_global_ind2val(sqlContext, slates, data_dir, start_date, drop_groups=False, min_item_clicks=2, min_user_clicks=10):

    ### ITEMS
    active_items = (
        #slates.select(F.explode('action').alias('id'))
        slates.select(F.col("click").alias("id"))
        .groupby('id').count()
        .withColumnRenamed(
            'count', 'actions')
        .filter(F.col('actions') > min_item_clicks)
        .filter(F.col('id') != 'noClick'))#.persist()

    #%%
    # Fetch all items published 90 days before start_date:
    logging.info("Get items from contentDB..")
    published_after = start_date-datetime.timedelta(120)

    query_pars = [
        'vertical', 'main_category', 'sub_category', 'prod_category', 'make',
        'county', 'model'
    ]

    category_pars = ['vertical', 'main_category','county'] #, #'sub_category', 'prod_category', 'make', 'model','municipality','city_area']

    # Build alternative content based on text string:
    text_query_pars = [
        'vertical', 'main_category', 'sub_category', 'prod_category', 
        'municipality', 'county', "district", 'heading'
    ]
    textprior_sql = " , ".join([f"' <{key.upper()}> ', {key}"for key in text_query_pars])

    q = f"""
        select id, {', '.join(query_pars)},
        concat({textprior_sql}) as textprior
        from ad_content
        join post_code on ad_content.post_code = post_code.code
        where (published >= '{published_after}')
        """
        #        AND state = 'ACTIVATED'


    content = (
        FINNHelper.contentdb(sqlContext, q).coalesce(200)
        # CONCAT CATEGORY STRINGS
        .fillna('', subset=category_pars).dropDuplicates(['id'])
        .withColumn('contentDB', F.lit(True))
        )

    items = (
        active_items
        .join(content, on='id',how='inner')
        )

    ## BUILD CATEGORY STRUCTURE:
    df = items.withColumn('category', F.col(category_pars[0]))
    if not drop_groups:
        for catvar in category_pars[1:]:  # skip first (vertical)
            df = add_catvar_if_above_threshold(df=df, catvar=catvar, th=200)
            #logging.info(f"{catvar} : {df.count()}")

    items = df.drop(*category_pars)
    
    items_loc = items.toPandas()
    logging.info(f'After filters we are left with {items_loc.shape[0]} items.')

    ### CONCAT DUMMYITEMS WITH REAL ITEMS:
    # Create some dummyitems
    unk = '<UNK>'
    fillitems = pd.DataFrame({
        'id': ['PAD', 'noClick', unk],
        'idx': [0, 1, 2],
        'actions': [-1, -1, -1],
        'category': ['PAD', 'noClick', unk],
        'textprior': ['PAD', 'noClick', unk],
    })
    # Add index to all real items (starting at 3):
    items_loc['idx'] = range(3, items_loc.shape[0] + 3)
    all_items = pd.concat([fillitems,
                           items_loc]).reset_index(drop=True).fillna(unk)

    if drop_groups:
        logging.info('Dropping group information from datatset...')
        all_items['category'] = unk

    ind2val = {}
    ind2val['itemId'] = {
        int(idx): str(item)
        for idx, item in zip(all_items.idx.values, all_items.id.values)
    }

    ## ATTRIBUTE VECTORS
    # Attribute vectors on items. Each index of the array has a value corresponding
    # to the item index as described in ind2val['itemId]
    itemattr = {}
    # actions
    actions = np.zeros((all_items.idx.shape))
    for idx, action in zip(all_items.idx.values, all_items.actions.values):
        actions[idx] = action
    itemattr['actions'] = actions

    # textpriors:
    textpriors = np.zeros((all_items.idx.shape), dtype=object)
    for idx, text in zip(all_items.idx.values, all_items.textprior.values):
        textpriors[idx] = text
    itemattr['textpriors'] = textpriors

    # Categorical variables:
    for var in ['category']:
        ind2val[var] = {
            int(idx): str(item)
            for idx, item in zip(all_items.idx.values, all_items[var].values)
        }
        ind2val['category'] = {
            idx: name
            for idx, name in enumerate(all_items['category'].unique())
        }
        vec = np.zeros((all_items.idx.shape))
        val2ind = {val: idx for idx, val in ind2val[var].items()}
        for idx, item in zip(all_items.idx.values, all_items[var].values):
            vec[idx] = int(val2ind.get(item))
        itemattr[var] = vec

    # displayType
    display_types = slates.select("displayType").distinct().toPandas().values.flatten()
    ind2val['displayType'] = {i+1 : val for i, val in enumerate(display_types)}
    ind2val['displayType'][0] = "<UNK>"
    ## USERS

    ## Prepare Users
    user_table = (
        slates
        .groupby('userId')
        .agg(
            F.count('*').alias('tot_clicks'),
            )
        .filter(F.col('tot_clicks') >= min_user_clicks)
        .select('userId')
        )

    unique_users = user_table.toPandas().values.flatten()
    ind2val['userId'] = {i+1 : val for i, val in enumerate(unique_users)}
    ind2val['userId'][0] = "<UNK>"
    logging.info(f'There are {len(ind2val["userId"])} users in the dataset.')

    ## SAVE ind2val and attributes:
    with open(f'{data_dir}/ind2val.pickle', 'wb') as handle:
        pickle.dump(ind2val, handle)
    logging.info('saved ind2val.')

    with open(f'{data_dir}/itemattr.pickle', 'wb') as handle:
        pickle.dump(itemattr, handle)
    logging.info('saved itemattr.')
    return ind2val, itemattr, user_table


def prepare_sequences(sqlContext,
                      slates,
                      ind2val,
                      data_dir,
                      maxlen_time,
                      maxlen_action,
                      limit=False):

    logging.info('Indexize all slates with ind2val...')

    item2ind = {val: ind for ind, val in ind2val['itemId'].items()}
    # Indexize click column:
    indexize_click = F.udf(lambda x: item2ind.get(x,2), IntegerType())

    # Indexize action for all dataset AND ADD NOCLICK ACTION IN FIRST ELEMENT
    def indexize_action_array(L):
        if L is None:
            return None
        if len(L) >= 0:
            return [1] + [int(item2ind.get(l, 2)) for l in L[:maxlen_action]]
        else:
            return None
    indexize_action_array = F.udf(indexize_action_array, ArrayType(IntegerType()))

    # Indexize displayType for all dataset:
    displaytype2ind = {val: ind for ind, val in ind2val['displayType'].items()}
    indexize_displayType = F.udf(lambda x: displaytype2ind.get(x,0), IntegerType())

    ##% INDEXIZE userId
    userId2ind = {val: ind for ind, val in ind2val['userId'].items()}
    indexize_userId = F.udf(lambda x: userId2ind.get(x,0), IntegerType())

    #%% Indexize all entries to integer indicies:
    fulldat = (
        slates
        .withColumn("userId", indexize_userId("userId"))
        .withColumn('click', indexize_click('click'))
        .withColumn('action', indexize_action_array('action'))
        .withColumn("displayType", indexize_displayType("displayType"))
        )
    
    # COMPUTE CLICK INDEX

    def first_idx(L, val):
        for i, l in enumerate(L):
            if l == val:
                return i
        return None
    first_idx = F.udf(first_idx, IntegerType())
    
    fulldat = (
        fulldat
        .filter(F.col("click") != 2) # filter UNK clicks
        .withColumn("click_idx", first_idx("action","click"))
        .filter(F.col("click_idx").isNotNull()) # Remove slates that had clicks but the item is not in inscreen
    )

    logging.info("Collect slates to sequences..")
    w = Window.partitionBy('userId').orderBy('timestamp')

    df = fulldat
    columns = ["displayType", "timestamp", "action", "click", "click_idx"]

    for col in columns:
        df = df.withColumn(col, F.collect_list(col).over(w))

    sequence_slates = (
        df.groupby("userId")
            .agg(*[F.max(col).alias(col) for col in columns])
    )

    ## -- COLLECT AND SAVE
    logging.info('starting collect spark2pandas..')
    dat = sequence_slates.toPandas()

    logging.info(f'Dataset has {dat.shape[0]} sequences. Further, it has the following unique indicies:')
    for key, val in ind2val.items():
        logging.info(f"{key} : {len(val)}")
    
    logging.info("processing data to tensors..:")
    data = construct_data_torch_tensors(dat, maxlen_time=maxlen_time, maxlen_action=maxlen_action)
    logging.info('Save data to files..')
    if not limit:
        save_dir = f'{data_dir}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        logging.info('starting saving..')

        # All data
        torch.save(data, f'{save_dir}/data.pt')
        logging.info('saved data.')
    else:
        logging.info('Limit was set, skip saving..')
    logging.info(f'Done preparing and saving dataset.')
    return True


# %%
def construct_data_torch_tensors(dat, maxlen_time, maxlen_action):
        logging.info(
            f'Building dataset of {dat.shape[0]} sequences. (timelength, candlength) = ({maxlen_time}, {maxlen_action})'
        )
        dat = dat.reset_index(drop=True)

        action = torch.zeros(
            (len(dat), maxlen_time,
             maxlen_action)).long()  # data_sequence, time_seq, candidates
        click =       torch.zeros(len(dat), maxlen_time).long()  # data_sequence, time_seq
        displayType = torch.zeros(len(dat), maxlen_time).long()  # data_sequence, time_seq

        click_idx = torch.zeros(
            len(dat), maxlen_time).long()  # data_sequence, time_seq
        lengths = torch.zeros((len(dat), maxlen_time)).long()

        userId = torch.tensor(dat.userId.values)

        for i in dat.index:
            # action
            row_action = dat.at[i, 'action'][:maxlen_time]
            obs_time_len = min(maxlen_time, len(row_action))

            lengths[i, :obs_time_len] = torch.tensor(
                [len(l) for l in row_action])

            row_action_pad = torch.from_numpy(
                pad_sequences(row_action[:obs_time_len],
                              maxlen=maxlen_action,
                              padding='post',
                              truncating='post'))
            action[i, :obs_time_len] = row_action_pad

            # Click
            click[i, :obs_time_len] = torch.tensor(
                dat.at[i, 'click'])[:obs_time_len]

            # Click index
            click_idx[i, :obs_time_len] = torch.tensor(
                dat.at[i, 'click_idx'])[:obs_time_len]

            displayType[i,:obs_time_len] = torch.tensor(dat.at[i, 'displayType'])[:obs_time_len]

        ## Set those clicks that were above the maximum candidate set to PAD:
        logging.info(
            f'There are {(click_idx >= maxlen_action).float().sum()} clicks that are above the maxlength action. Setting to click_idx=0 but with click= 0 ("PAD")..'
        )
        click_idx[(click_idx >= maxlen_action)] = 0
        click[(click_idx >= maxlen_action)] = 0

        data = {
            'userId' : userId,
            'lengths': lengths,
            'displayType' : displayType,
            'action': action,
            'click': click,
            'click_idx': click_idx
        }
        return data

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


def prepare_dataset(data_dir, sample_uniform_action):
    logging.info(f'Building dataset for {data_dir}.')
    logging.info('Load ind2val..')
    with open(f'{data_dir}/ind2val.pickle', 'rb') as handle:
        ind2val = pickle.load(handle)

    logging.info('Load data..')
    data = torch.load(f'{data_dir}/data.pt')

    dataset = SequentialDataset(data, sample_uniform_action)

    with open(f'{data_dir}/dataset.pickle', 'wb') as handle:
        pickle.dump(dataset, handle, protocol=4)


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

def main_prepare(param):
    sc, sqlContext = FINNHelper.create_spark_cluster(driver_memory='200G',
                                                     max_result_size='16G')
    torch.set_grad_enabled(False)
    

    end_date = param.get('end_date', datetime.datetime.today().date())
    start_date = end_date - datetime.timedelta(param.get('lookback'))


    logging.info('-' * 20)
    logging.info('PREPARE DATASET..')
    logging.info('-' * 20)
    logging.info(f'Data period: \t [{start_date}, {end_date})')

    # make dir if not exist
    mkdir(f'{param.get("data_dir")}')
    ## FIND IND2VAL FOR ITEMS AND THE USERS THAT SHOULD BE INCLUDED
    logging.info("Read slates..")

    lake = sqlContext.read.parquet("gs://finn-slates/lake")
    if param.get("include_stream"):
        lake = lake.unionAll(sqlContext.read.parquet("gs://finn-slates/stream"))

    slates = (
        lake
        .withColumnRenamed('inscreen', 'action')
        .filter(F.col('date') <= end_date)  # remove future data
        .filter(F.col('date') >= start_date)  # remove old data
        # TMP FILTER TO SPEED UP:
        ##
        .withColumn("click", F.element_at("click", 1)) # Use only first click
        .fillna({'click' : "noClick"})
    )


    # Subsample the noclicks uniformly to get a more "balanced" dataset.
    subsample_noClick = param['subsample_noClick']
    if 0 < subsample_noClick < 1:
        logging.info(f"Subsample noClicks: Keep {subsample_noClick} of total noClicks in dataset..")
        slates = (
            slates.withColumn("isnoclick", F.col("click") =="noClick")
                .sampleBy("isnoclick", fractions={True: subsample_noClick, False: 1.0})
                .drop("isnoclick")
                .withColumn("click", F.coalesce(F.col("click"), F.lit("noClick")))
        )
    elif subsample_noClick == 1:
        slates = slates.withColumn("click", F.coalesce(F.col("click"), F.lit("noClick")))
    else:
        slates = slates.filter(F.col("click").isNotNull())

    slates = slates.coalesce(200)

    ## Prepare item indicies
    logging.info('Prepare ind2item and item attributes..')
    ind2val, itemattr, user_table = build_global_ind2val(sqlContext,
                                             slates=slates,
                                             start_date =start_date,
                                             data_dir=param.get('data_dir'),
                                             drop_groups=param.get('drop_category', False),
                                             min_item_clicks = param.get("min_item_clicks", 3),
                                             min_user_clicks = param.get("min_user_clicks", 3))


    logging.info('Starting on the sequences..')
    slates = slates.join(user_table, on='userId', how='inner') # filter to only "active enough" users
    ## PREPARE DATASETS

    logging.info('-- Prepare sequences..')
    maxlen_action=param.get('maxlen_action')
    maxlen_time=param.get('maxlen_time')
    prepare_sequences(sqlContext,
                      slates=slates,
                      ind2val=ind2val,
                      data_dir=param.get('data_dir'),
                      maxlen_time=maxlen_time,
                      maxlen_action=maxlen_action
    )

    logging.info('Done prepare.py')

#%%

def main_load_from_preproc_job(param):
    model_path = f'recommendations-models/TF/{param.get("load_data_from_preproc_job")}'
    DATAHelper.download_files_from_path(model_path, files="*", local_path="data")

#%% load_previous_checkpoint
def download_prev_checkpoint(param):
    # If its only set to true, find model name by directory:
    if param.get("load_previous_checkpoint") == True:
        param["load_previous_checkpoint"] = os.getcwd().split("/")[-1]
    
    model_path = f'recommendations-models/TF/{param.get("load_previous_checkpoint")}'
    DATAHelper.download_files_from_path(
        model_path, 
        files=['*.pyro','ind2val.pickle', "itemattr.pickle", "*tfevents*"], 
        local_path="old_checkpoint",
        epoch=param.get("earliest_checkpoint"))


def load_prev_checkpoint_reindex_and_set_pyro_state(param, ind2val, itemattr):
    # Load old state:
    old_model_name =[f for f in os.listdir("old_checkpoint") if ".pyro" in f][0]
    old_model_path = f"old_checkpoint/{old_model_name}"
    old_state = torch.load(open(old_model_path,"rb"), map_location=param.get("device"))
    old_ind2val = pickle.load(open("old_checkpoint/ind2val.pickle", "rb"))
    pyro.get_param_store().set_state(old_state)

    # GROUP VECTORS:
    old_item2ind = {item: idx for idx, item in old_ind2val['category'].items()}
    item2ind = {val: key for key, val in ind2val['category'].items()}
    for parname in ['item_model.groupvec.weight','item_model.groupscale.weight']:
        for mode in ['mean','scale']:
            par = f'{parname}-{mode}'
            try:
                emb = PYTORCHHelper.remap_embedding(
                    old_embedding= pyro.param(par),
                    new_embedding= 0.01 + torch.zeros((param['num_groups'], param['item_dim'])),
                    old_lookups=old_item2ind,
                    new_lookups=item2ind)
                
                pyro.get_param_store().__setitem__(
                    name=par, 
                    new_constrained_value= torch.tensor(emb, requires_grad=True).to(param.get("device"))
                    )
            except Exception as e:
                logging.info(f"Could not set state on {par}. Error:")
                print(e)

    # ITEM VECTORS:
    # initialize new embedding with groups:
    old_item2ind = {item: idx for idx, item in old_ind2val['itemId'].items()}
    item2ind = {val: key for key, val in ind2val['itemId'].items()}
    for mode in ['mean','scale']:
        par = f'item_model.itemvec.weight-{mode}'
        emb = PYTORCHHelper.remap_embedding(
            old_embedding= pyro.param(par),
            new_embedding= pyro.param(f"item_model.groupvec.weight-{mode}")[itemattr['category']],
            old_lookups=old_item2ind,
            new_lookups=item2ind)

        pyro.get_param_store().__setitem__(
            name=par, 
            new_constrained_value= torch.tensor(emb, requires_grad=True).to(param.get("device"))
            )


    # Shrink all parameters to avoid overfitting:
    logging.info("shrinking checkpoint parameters")
    shrink_factors = {
        'item_model.groupscale.weight-mean' : 0.2,
        'item_model.itemvec.weight-scale' : 0.2
        }
    for par in pyro.get_param_store().get_all_param_names():
        factor = shrink_factors.get(par, 0.7)
        logging.info(f"shrinking {par} with {factor}.")
        pyro.get_param_store().__setitem__(
            name=par, 
            new_constrained_value= (pyro.param(par)*factor).detach().clone()
            )
    return True


#%%
if __name__ == '__main__':
    param = utils.load_param()
    if param.get("load_previous_checkpoint"):
        download_prev_checkpoint(param)
        
    if param.get("load_data_from_preproc_job"):
        main_load_from_preproc_job(param)
    else:
        main_prepare(param)

