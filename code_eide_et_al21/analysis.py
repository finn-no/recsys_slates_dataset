#%%
import pyro
from train import *
import train
import FINNPlot
def plot_itemidx_array(arr,nrow=None):
    if nrow is None:
        nrow = arr.size()[1]
    finnkoder = [ind2val['itemId'][r.item()] for r in arr.flatten()]
    return FINNPlot.add_image_line(finnkoder, nrow=nrow)

param, ind2val, itemattr, dataloaders, sim = train.load_data()
model, guide = train.initialize_model(param,ind2val, itemattr, dataloaders)
param, ind2val, trainer = train.train(param, ind2val, itemattr, dataloaders, sim, model, guide, run_training_loop=False)

#%%
pyro.clear_param_store()
m = "Jon-Arya-gru-hier-clip1000:lik=0.026:gui=7.18366887024712:lea=0.000434:pri=7.14:pri=0.01:pri=0.01:na" # -20M-steps
pyro.get_param_store().load(f"checkpoints/{m}.pyro", map_location=param['device'])
guide.initialize_parameters()
#%%
dl = iter(dataloaders['train'])
dummybatch = next(dl)
dummybatch['phase_mask'] = (dummybatch['mask_type']==1).float()
dummybatch = {key: val.long().to(param.get("device")) for key, val in dummybatch.items()}

#%% Distirbution of click probabilities over slate at a given time t for one user
#h0batch_fixed = par['h0-batch']
for i in range(10):
    par=guide(dummybatch)
    #par['h0-batch'] = h0batch_fixed
    res = model.likelihood(dummybatch, par=par)
    #res_prior = model._compute(dummybatch)
    b = 1
    t = 4
    scores = res['score'][b,t,]
    mask = scores>-100
    plt.plot((scores.exp()/(scores[mask].exp().sum()))[mask].cpu().detach())

#%%
#model(dummybatch)
par=guide(dummybatch)

res = model.likelihood(dummybatch, par=par)
_ = plt.plot((res['zt']**2).sum(0).detach().cpu())
plt.show()
plt.plot(res['zt'].std((0,1)).detach().cpu())

#%% ADD EMBEDDINGS
trainer.step=2
from bayesrec import pyrotrainer
emb = pyrotrainer.VisualizeEmbeddings(ind2val=ind2val)
emb(trainer)
# %% ANALYSE ITEMS

#%% How large variance is there on items:
V_scale = pyro.param("model.item_model.itemvec.weight-scale").detach().cpu()#.numpy()
V_scale_norm = V_scale.mean(1).numpy()
_ = plt.hist(V_scale_norm, bins = 100)
#%% Number of exposures on each item
plt.show()
popvec = torch.zeros((len(V_scale_norm),))
id, cnts = dataloaders['train'].dataset.data['action'].unique(return_counts=True)
for i, cnt in zip(id, cnts):
    popvec[i] = cnt
popvec[:3] = 1

popvec = popvec.detach().cpu().numpy()
#%%
_ = plt.hist(popvec, bins = 100, range=(0,1000))
plt.yscale("log")
#%%
cat_idx, cat_cnts=np.unique(itemattr['category'], return_counts=True)
catidx2catcount = {idx : count for idx, count in zip(cat_idx, cat_cnts)}


#%% dist groupvec to item vec
V = pyro.param("model.item_model.itemvec.weight-mean").detach()#.cpu()#.numpy()
Vg = pyro.param("model.item_model.groupvec.weight-mean").detach()#.cpu()#.numpy()
item_groupvec = Vg[itemattr['category']]
dist_itemvec_groupvec = ((V-item_groupvec)**2).sum(1).sqrt()
dist_itemvec_groupvec = dist_itemvec_groupvec.cpu().numpy()

#%%
#%% dist SCALE groupvec to item vec
V = pyro.param("model.item_model.itemvec.weight-scale").detach()#.cpu()#.numpy()
Vg = pyro.param("model.item_model.groupscale.weight-mean").detach()#.cpu()#.numpy()
item_groupvec = Vg[itemattr['category']].abs()
dist_scale_itemvec_groupvec = ((V-item_groupvec)**2).sum(1).sqrt()
dist_scale_itemvec_groupvec = dist_scale_itemvec_groupvec.cpu().numpy()


#%%

import seaborn as sns
import pandas as pd
df = pd.DataFrame({
    'Number of Views' : popvec+1,
    'Average posterior scale of item vector' : V_scale_norm,
    'category' : np.array([ind2val['category'][i][:3] for i in itemattr['category']]),
    'category_count' : np.array( [ catidx2catcount[i]  for i in itemattr['category'] ] ),
    'Average distance between item vector and correspodning group vector' : dist_itemvec_groupvec,
    'dist_scale_itemvec_groupvec' : dist_scale_itemvec_groupvec
    })
#%% PLOT VARIOUS PLOTS WITH NUM_VIEWS IN X AXIS AND A VAR IN Y
Y_vars = [
    'Average posterior scale of item vector',
    'Average distance between item vector and correspodning group vector',
    ]
groups = df.category.unique()[3:]
df_loc = df[df.category.isin(groups)]

for i, y in enumerate(Y_vars):
    def hexbin(x, y, color, **kwargs):
        sns.set_style("white")
        cmap = sns.light_palette(color, as_cmap=True)
        plt.hexbin(x, y, gridsize=100, cmap=cmap,**kwargs)
    g = sns.FacetGrid(df_loc, height=5, aspect=2)
    g.map(hexbin, 'Number of Views', y, xscale="log", bins="log", yscale="log")
    plt.title(f"Number of Views vs. {y}")


#%% Plot num views vs avg_posterior variance per vertical
groups = df.category.unique()[3:]
df_loc = df[df.category.isin(groups)]

def hexbin(x, y, color, **kwargs):
    cmap = sns.dark_palette(color, as_cmap=True)
    plt.hexbin(x, y, gridsize=100, cmap=cmap,**kwargs)
g = sns.FacetGrid(df_loc, hue="category",col="category",size=5,col_wrap=3)
g.map(hexbin, "num_views", "avg_posterior_scale", xscale="log", bins="log")
#%% How much does item vectors deviate from their group?

# %%
#idx = 43
#temp = 0.0
import requests
seen_finncodes = ["190392685", "165940725"]
recs_from_prod = str(requests.get(f"https://www.finn.no/recommendation-viewer/predict?items={', '.join(seen_finncodes)}&vertical=multi&recommenderId=mul-bayesgru-hier40").content)

#%%

item2ind = {val : key for key, val in ind2val['itemId'].items()}
seen_idx = [item2ind[i] for i in seen_finncodes]
clickvec = torch.tensor(seen_idx).long().unsqueeze(0)

smallbatch = {
    'click' : clickvec.to(model.device)}

#smallbatch['click'][0,5:] = dummybatch['click'][idx+1,:5].unsqueeze(0).to(model.device)
useguide = lambda *args, **kwargs: guide(temp=0, *args, **kwargs)
recs = model.recommend(smallbatch, par=useguide, num_rec=10)
#%%
# Does anyone of the recommended here also being recommended in prod?
sum([ind2val['itemId'][r.item()] in recs_from_prod for r in recs.squeeze()])

#%%
plt.imshow(plot_itemidx_array(recs).permute(1,2,0))


views = smallbatch['click'].flatten()
num_recs=5
num_time=2
M = torch.zeros(num_recs+1, num_time)
M[0,:] = views[:num_time]
for t_rec in range(num_time):
    M[1:,t_rec] = model.recommend(smallbatch, par=lambda *args, **kwargs: guide(temp=temp, *args, **kwargs), num_rec=num_recs, t_rec=t_rec)

plt.figure(figsize=(30,30))
plt.imshow(plot_itemidx_array(M).permute(1,2,0))

#%%
def get_count_vector(label="click"):
    id, cnts = dataloaders['train'].dataset.data[label].unique(return_counts=True)
    clickvec = torch.zeros((len(ind2val['itemId']),))
    for i, cnt in zip(id, cnts):
        clickvec[i] = cnt
    return clickvec
    
#%%
clickvec = get_count_vector("click")
viewvec = get_count_vector("action")

viewvec[viewvec==0] = 1e-6
popvec = clickvec/viewvec
popvec[viewvec<1000]=0
#%%
most_clicked_items = popvec.argsort()[-20:]
print(popvec[most_clicked_items])
print(max(popvec))
plt.imshow(plot_itemidx_array(most_clicked_items,nrow=10).permute(1,2,0))

# %%
