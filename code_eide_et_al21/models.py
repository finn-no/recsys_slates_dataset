#%%
import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro import optim
from pyro.infer import SVI, Trace_ELBO
from pyro.nn import PyroSample, PyroModule, pyro_method
from pyro.infer.autoguide import AutoDiagonalNormal, init_to_sample, AutoDelta
from pyro import poutine
import torch.distributions.constraints as constraints
import logging
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
logging.basicConfig(format='%(asctime)s %(message)s', level='INFO')
import warnings

def score_dot(x,y, *args, **kwargs):
    return (x*y).sum(-1)
def score_cosine(x,y):
    score_dot(x,y) / (score_dot(x,x)*score_dot(y,y)).sqrt()

def score_euclidean(x,y, ynorm=None):
    #time.sleep(0.1)
    x_norm = (x**2).sum(-1)#.view(-1, 1)
    #print(x_norm.size())
    dot_last_dim = y.matmul(torch.transpose(x,-1,-2)).squeeze(dim=-1) #(x*y).sum(-1)
    dist = x_norm - 2.0 * dot_last_dim
    #print(dist.size())
    if ynorm is None:
        y_norm = (y**2).sum(-1)
        dist += y_norm
    else:
        dist += ynorm

    return -dist.clamp_min(1e-10).sqrt()

### WRAPPER FOR MODELS:
def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

class PyroRecommender(PyroModule):
    """ A Pyro Recommender Module """
    def __init__(self, **kwargs):
        super().__init__()
        self.param = kwargs # this is ugly, but needs to be done for now.

        # Register all input vars in module:
        for key, val in kwargs.items():
            setattr(self,key,val)

        self.initialize_parameters()

        # Set parameters
    def initialize_parameters(self):
            logging.info(f"Initializing model with usermodel={self.param.get('user_model_module')}, user_init={self.param.get('user_init_module')} itemmodel={self.param.get('item_model_module')}")
            self.item_model = item_models[self.param.get('item_model_module')](**self.param)
            self.user_init_model = user_init_models[self.param.get("user_init_module", "zero")](**self.param)
            self.user_model = user_models[self.param.get('user_model_module')](**self.param)
            self.scorefunc = scorefunctions[self.dist]

            # Common for all:
            if self.prior_softmax_mult_scale > 0:
                self.softmax_mult = PyroSample( prior = dist.Normal(torch.tensor(1.0), self.prior_softmax_mult_scale *torch.tensor(1.0)))
            else:
                self.softmax_mult = torch.tensor(1.0)
            self.bias_noclick = PyroSample(
                prior = dist.Normal(
                    torch.zeros((self.maxlen_action +1,)),
                    self.prior_bias_scale * torch.ones( (self.maxlen_action +1,))
                ))
    ## 
    def init_set_of_real_parameters(self, seed = 1):
        """ If the model is functioning as an environment this function will initialize it with the true parameters."""
        torch.manual_seed(seed)
        par_real = {}

        # Initialize common parameters:
        par_real['softmax_mult'] =  torch.tensor(self.true_softmax_mult).float()
        #par_real['gamma'] = torch.tensor(self.true_gamma)
        par_real['bias_noclick'] = self.true_bias_noclick * torch.ones( (self.maxlen_action +1,))

        # Initalize item parameters:
        par_real_item = self.item_model.init_set_of_real_parameters(true_softmax_mult=par_real['softmax_mult'],seed=seed)
        for key, val in par_real_item.items():
            par_real[f'item_model.{key}'] = val

        # Init user parameters:
        par_real_user = self.user_model.init_set_of_real_parameters(seed)
        for key, val in par_real_user.items():
            par_real[f'user_model.{key}'] = val

        # Init USER INIT parameters
        # Set user initial conditions to specific groups:
        self.user_init_group = torch.randint(self.num_groups, (self.num_users,))
        par_real['h0'] = par_real['item_model.groupvec.weight'][self.user_init_group]

        self.par_real = par_real

    def get_real_par(self, batch):

        """ Function that outputs a set of real parameters, including setting h0-batch which is batch specific"""
        out = self.par_real
        out['h0-batch'] = self.par_real["h0"][batch['userId']]
        return out
    
    @pyro_method
    def forward(self, *args, **kwargs):
        """ Need to abstract away using forward for the main logic so we can trace with .forward() in production afterwards"""
        return self._compute(*args, **kwargs)
    
    @pyro_method
    def _compute(self, batch, mode = "likelihood", t_rec=None):
        with poutine.scale(scale= (1/self.prior_temperature)): # scale all prior params here

            batch_size, t_maxclick = batch['click'].size()
            # sample item dynamics
            itemvec = self.item_model()
            

            # Sample user dynamic parameters:
            softmax_mult = self.softmax_mult.abs()
            bias_noclick = self.bias_noclick
            
            h0 = self.user_init_model(batch)
            Z = self.user_model(batch, V=itemvec, h0=h0)

            # NB: DOES NOT WORK FOR LARGE BATCHES:
            if mode =="userprofile":
                return Z

            target_idx = batch['click_idx']

            lengths = (batch['action'] != 0).long().sum(-1)
            action_vecs = self.item_model(batch['action']) #itemvec[batch['action']]

            # Compute scores for all actions by recommender
            scores = self.scorefunc(Z[:,:t_maxclick].unsqueeze(2), action_vecs)

            scores = scores*softmax_mult

            # PAD MASK: mask scores through neg values for padded and UNK candidates:
            scores[batch['action'] <= 2] = -100

            # Add a constant based on inscreen length to the no click option:
            batch_bias = bias_noclick[lengths]
            batch_bias = (batch['action'][:, :, 0] == 1).float() * batch_bias
            scores[:,:,0] = batch_bias
            scores = scores.clamp(-100,20)

        # SIMULATE
        # If we want to simulate, then t_maxclick = t_maxclick+1
        if mode == "simulate":
            # Check that there are not click in last time step:
            if bool((batch['click'][:,-1] != 0 ).any() ):
                warnings.warn("Trying to sample from model, but there are observed clicks in last timestep.")
            
            gen_click_idx = dist.Categorical(logits=scores[:, -1, :]).sample()
            return gen_click_idx
        
        if mode == "likelihood":
            # MASKING
            time_mask = (batch['click'] != 0)*(batch['click'] != 2).float()
            mask = time_mask*batch['phase_mask']
            with poutine.scale(scale= (1/self.likelihood_temperature)):
                with pyro.plate("data", size = self.num_users, subsample = batch['userId']):
                    obsdistr = dist.Categorical(logits=scores).mask(mask).to_event(1)
                    pyro.sample("obs", obsdistr, obs=target_idx)
            
        return {'score' : scores, 'zt' : Z, 'V' : itemvec}

    @pyro_method
    def predict(self, batch, t_rec=-1, production=False):
        """
        Computes scores for each user in batch at time step t.
        NB: Not scalable for large batch.
        
        batch['click_seq'] : tensor.Long(), size = [batch, timesteps]
        """
        Z = self._compute(batch, mode = "userprofile")

        if production is True:
            itemvec = self.item_embedding.weight
            itemnorm = self.item_embedding.bias.unsqueeze(0)
        else:
            itemvec = self.item_model()
            itemnorm=None

        zt_last = Z[:, t_rec, ]
        score = self.scorefunc(zt_last.unsqueeze(1), itemvec.unsqueeze(0), itemnorm)
        return score, Z

    ## GENERAL FUNCTIONS ###
    @pyro_method
    def simulate(self, batch):
        return pyro.condition(
            lambda batch: self._compute(batch, mode="simulate"),
            data=self.get_real_par(batch))(batch)

    @pyro_method
    def likelihood(self, batch, par = None): 
        if par is None:
            par = self.get_real_par(batch)

        return pyro.condition(
            lambda batch: self._compute(batch, mode="likelihood"),
            data=par)(batch)

    @pyro_method
    def predict_cond(self, batch, par=None, **kwargs):
        """
                Par can either be the string "real" or a function that outputs the parameters when called with a batch of data.
        """
        if par is "real":
            par = self.get_real_par(batch)
        else:
            par = par(batch)
        return pyro.condition(
            lambda batch: self.predict(batch, **kwargs),
            data=par)(batch)

    @pyro_method
    def recommend(self, batch, num_rec=1, chunksize=3, t_rec=-1, par=None, **kwargs):
        """
        Compute predict & rank on a batch in chunks (for memory)
        Par can either be the string "real" or a function that outputs the parameters when called with a batch of data.
        """

        click_seq = batch['click']
        topk = torch.zeros((len(click_seq), num_rec), device=self.device)

        i = 0
        for batch_chunk in dict_chunker(batch, chunksize):
            
            pred, ht = self.predict_cond(
                batch=batch_chunk, t_rec=t_rec, par=par)
            
            topk_chunk = 3 + pred[:, 3:].argsort(dim=1,
                                                 descending=True)[:, :num_rec]
            topk[i:(i + len(pred))] = topk_chunk
            i += len(pred)
        return topk

    @pyro_method
    def recommend_inslate(self, batch, num_rec=1, chunksize=3, t_rec=-1, par=None, **kwargs):
        """
        Inslate thompson recommendation.
        Compute predict & rank on a batch in chunks (for memory)
        Par can either be the string "real" or a function that outputs the parameters when called with a batch of data.
        """
        click_seq = batch['click']
        topk = torch.zeros((len(click_seq), num_rec), device=self.device)

        i = 0
        for click_chunk, userId in zip(
            chunker(click_seq, chunksize),
            chunker(batch['userId'], chunksize)):

            chunklen = len(click_chunk)
            # Sample many posterior draws and collect topK:
            topk_samples = torch.zeros((chunklen,num_rec,num_rec))
            for s in range(num_rec):
                pred, ht = self.predict_cond(
                    batch={'click' : click_chunk, 'userId' : userId}, t_rec=t_rec, par=par)
                topk_samples[:,:,s] = 3 + pred[:, 3:].argsort(dim=1,
                                                        descending=True)[:, :num_rec]

            # Flatten all topKs per users into a priorized list of shape [num_user, num_rec*num_rec]:
            priority = topk_samples.view(chunklen, -1)

            # Take the num_rec top unique items and return topk_chunk = [num_user, num_rec]
            topk_chunk = torch.zeros((chunklen, num_rec))
            for u in range(chunklen):
                offset=0
                for k in range(num_rec):
                    # If item already exist, offset by one:
                    while priority[u,k+offset] in topk_chunk[u]:
                        offset += 1
                    topk_chunk[u,k] = priority[u,k+offset]

            # Fill into the general recommendation matrix:
            topk[i:(i + len(pred))] = topk_chunk
            i += len(pred)
        return topk
    
    ## PRODUCTION FUNCTIONS ##
    def predict_production(self, click_seq):
        padded_click_seq = torch.cat( (self.padding, click_seq[0]))[-self.maxlen_time:].unsqueeze(0)

        if self.h0_amort:
            h0_mean, h0_scale = self.h0_amortization_func(
                batch_click=padded_click_seq, **self.amort_params)
            if self.h0_temperature:
                self.sampled_par['h0-batch'] = torch.distributions.Normal(h0_mean, self.h0_temperature*h0_scale).sample()
            else:
                self.sampled_par['h0-batch'] = h0_mean
            
        with pyro.condition(data=self.sampled_par):
            batch = {'click' : padded_click_seq}
            score, Z = self.predict(batch, t_rec=-1,production=True)
            return score.squeeze()

    @torch.no_grad()
    def prepare_model_for_production(self, guide, sampled_par, h0_temperature=None):
        """ All steps necessary post-training to make model traceable. NB: Changes the self object and is non-reversible"""

        self.h0_temperature = h0_temperature
        logging.info(f"Setting h0 temperature = {self.h0_temperature}")

        if self.h0_amort:
            self.h0_amortization_func = guide.h0_amortization_func
            self.amort_params = {
                'V' : pyro.param("model.item_model.itemvec.weight-mean").detach(),
                'amort_scalefactor' : pyro.param("h0-amort-scalefactor").detach()
            }
        self.sampled_par = sampled_par
        # Prepare twoemb vector:
        self.item_embedding = nn.Linear(self.item_model.itemvec.weight.size()[0], self.item_model.itemvec.weight.size()[1])
        self.item_embedding.weight.data = self.item_model().clone().detach()
        self.item_embedding.bias.data  = (self.item_embedding.weight**2).sum(1).clone().detach() # Compute y norm
        
        # padding vector:
        self.padding = torch.zeros(self.maxlen_time).long()

        self.forward = self.predict_production

    def visualize_item_space(self):
        if self.item_dim ==2:
            V = self.par_real['item_model.itemvec.weight'].cpu()
            sns.scatterplot(V[:, 0].cpu(),
                            V[:, 1].cpu(),
                            hue=self.item_group.cpu())
            plt.xlim(-2, 2)
            plt.ylim(-2, 2)
            plt.show()
        if self.item_dim == 3:
            visualize_3d_scatter(self.par_real['item_model.itemvec.weight'])

def dict_chunker(dict_of_seqs, size):
    "Iterates over the first dimension of a dict of sequences"
    length = len(dict_of_seqs[list(dict_of_seqs.keys())[0]]) # length of first idex
    return ( {key : seq[pos:pos + size] for key, seq in dict_of_seqs.items()} for pos in range(0, length, size))

## USER INIT MODULES

class UserInitNormal(PyroModule):
    """ Returns an initial value of zero for all users """
    def __init__(self, num_users, hidden_dim, prior_userinit_scale, **kwargs):
        super().__init__()
        self.prior_userinit_scale = prior_userinit_scale
        self.num_users = num_users
        self.hidden_dim = hidden_dim

    def forward(self, batch, **kwargs):
        batch_size, t_maxclick = batch['click'].size()
        #with pyro.plate("user-init-plate", size = self.num_users, subsample = batch['userId']):
        # Sample initial hidden state of users:
        h0 = pyro.sample("h0-batch",
        dist.Normal(
            torch.zeros((batch_size, self.hidden_dim)), 
            self.prior_userinit_scale*torch.ones((batch_size, self.hidden_dim))
            ).to_event(1)
            )
        return h0

class UserInitCompactNormal(PyroModule):
    """ Returns an initial value of zero for all users """
    def __init__(self, num_users, init_dim, hidden_dim, prior_userinit_scale, **kwargs):
        super().__init__()
        self.prior_userinit_scale = prior_userinit_scale
        self.num_users = num_users
        self.hidden_dim = hidden_dim
        self.init_dim = init_dim
        nn.Linear
        self.init2hidden = build_linear_pyromodule(in_features= self.init_dim, out_features=self.hidden_dim, bias=False)

    def forward(self, batch, **kwargs):
        batch_size, t_maxclick = batch['click'].size()
        #with pyro.plate("user-init-plate", size = self.num_users, subsample = batch['userId']):
        # Sample initial hidden state of users:
        h0 = pyro.sample("h0-batch",
        dist.Normal(
            torch.zeros((batch_size, self.init_dim)), 
            self.prior_userinit_scale*torch.ones((batch_size, self.init_dim))
            ).to_event(1)
            )
        return self.init2hidden(h0)

class UserInitZero(PyroModule):
    """ Returns an initial value of zero for all users """
    def __init__(self, hidden_dim, **kwargs):
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, batch, **kwargs):
        batch_size, t_maxclick = batch['click'].size()
        h0 = torch.zeros((batch_size, self.hidden_dim))
        return h0

class UserLinearGru(PyroModule):
    def __init__(self, *args,**kwargs):
        super().__init__()
        logging.info("Initializing LinearGru combo user model")
        self.gru = UserGRU(*args,**kwargs)
        self.linear = UserLinear(*args, **kwargs)
    
    def forward(self, *args, **kwargs):
        return 0.5*(self.gru(*args, **kwargs) + self.linear(*args, **kwargs))



class UserLinear(PyroModule):
    """ 
    H_t+1 = gamma*H_t + (1-gamma)*v_{c_{t-1}^u}
    Global gamma that defines how much a user should remember of previous state and how much the clicked item should impact.
    """
    def __init__(
        self,
        num_users,
        prior_userinit_scale,
        device="cpu", 
        true_gamma=None,
        **kwargs,
        ):
        super().__init__()
        self.num_users = num_users
        self.prior_userinit_scale = prior_userinit_scale
        self.device = device
        self.true_gamma = true_gamma
        # Parameters:
        logging.info(f"Initializing Linear User with gamma prior Normal(0.5,0.3)")
        self.gamma = PyroSample( dist.Normal(torch.tensor( 0.5),torch.tensor(0.3)) )

    def init_set_of_real_parameters(self, seed = 1):
        par_real = {}
        par_real['gamma'] = torch.tensor(self.true_gamma)
        return par_real

    def forward(self,batch, V, h0, **kwargs):
        batch_size, t_maxclick = batch['click'].size()
        item_dim = V.size()[1]
        click_vecs = V[batch['click']]
        gamma = self.gamma
        gamma_batch = gamma*torch.ones_like(batch['click'])#.unsqueeze(-1)
        gamma_batch[( batch['click'] < 3 )] = 1.0 # set dummy gamma to 1 if no click, pad or unk

        H_list = [h0]
        for t in range(t_maxclick):
            gamma_batch_t = gamma_batch[:,t].unsqueeze(-1)
            h_new = gamma_batch_t * H_list[-1] + (1-gamma_batch_t)*click_vecs[:,t]

            

            H_list.append(h_new)

        H = torch.cat( [h.unsqueeze(1) for h in H_list], dim =1)
        return H

class UserMarkov(PyroModule):
    """ 
    Simple model that assumes that the user jumps to the location of the last clicked item at every time step.
    If the user does not click at a time step, the state does not change.
    """
    def __init__(
        self,
        device="cpu", 
        **kwargs,
        ):
        super().__init__()
        self.device = device

    def init_set_of_real_parameters(self, seed = 1):
        return {}
    def forward(self,batch, V, *args, **kwargs):

        batch_size, t_maxclick = batch['click'].size()
        click_vecs = V[batch['click']]
        
        # Force use zero vector for first interaction:
        h0 = torch.zeros((batch_size, item_dim)).float()
        
        H_list = [h0]
        for t in range(t_maxclick):
            gamma_batch = torch.zeros_like(batch['click'][:,t]).unsqueeze(-1)
            gamma_batch[(batch['click'][:,t]<3)] = 1.0 # set dummy gamma to 1

            h_new = gamma_batch * H_list[-1] + (1-gamma_batch)*click_vecs[:,t]
            H_list.append(h_new)

        H = torch.cat( [h.unsqueeze(1) for h in H_list], dim =1)
        return H

class UserGRU(PyroModule):
    """ 
    H_t+1 = GRU(H_t, v_{c_{t-1}^u} )
    """
    def __init__(
        self,
        item_dim,
        hidden_dim,
        num_users,
        prior_userinit_scale,
        prior_rnn_scale,
        device="cpu", 
        **kwargs,
        ):
        super().__init__()
        self.item_dim = item_dim
        self.hidden_dim = hidden_dim
        self.num_users = num_users
        self.prior_userinit_scale = prior_userinit_scale
        self.prior_rnn_scale = prior_rnn_scale
        self.device = device
        # Parameters:
        self.rnn = Gru(
            input_size=self.item_dim,
            hidden_size=self.hidden_dim,
            bias=False)
        set_noninform_prior(self.rnn, scale = self.prior_rnn_scale)

    def forward(self,batch, V, h0):
        batch_size, t_maxclick = batch['click'].size()
        click_vecs = V[batch['click']]
        Z_list = [self.rnn.hidden2output(h0)]
        h_new = h0
        for t in range(t_maxclick):
            h_old = h_new

            h_new = self.rnn(
                click_vecs[:,t], 
                h_old
                )
            # If dummy or noClick do not update h:
            stay_constant = (torch.lt( batch['click'][:,t], 3 )*1).unsqueeze(-1)
            h_new =  h_old*stay_constant + h_new*(torch.logical_not(stay_constant))
            
            output = self.rnn.hidden2output(h_new)

            Z_list.append(output)

        Z = torch.cat( [h.unsqueeze(1) for h in Z_list], dim =1)
        return Z

def build_linear_pyromodule(nn_mod = nn.Linear, scale = 1.0, eye=False, **kwargs):
    layer = PyroModule[nn_mod](**kwargs)
    set_noninform_prior(layer, scale =scale, eye=eye)
    return layer

def set_noninform_prior(mod, scale=1.0, eye=False):
    for key, par in list(mod.named_parameters()):
        mu = torch.zeros_like(par)
        if eye is True:
            min_dim = min(mu.size())
            mu[:min_dim, :min_dim] = torch.eye(min_dim)
        setattr(
            mod, key,
            PyroSample(
                dist.Normal(mu,
                            scale * torch.ones_like(par)).independent()))

class Gru(PyroModule):
    def __init__(self, input_size, hidden_size, bias=False, scale=1.0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.scale = scale
        self.W_z = build_linear_pyromodule(in_features=hidden_size, out_features= input_size, bias=self.bias, scale=self.scale, eye=False)

        self.W_ir = build_linear_pyromodule(in_features = input_size, out_features = hidden_size, bias=self.bias, scale=self.scale)
        self.W_id = build_linear_pyromodule(in_features = input_size, out_features = hidden_size, bias=self.bias, scale=self.scale)
        self.W_in = build_linear_pyromodule(in_features = input_size, out_features = hidden_size, bias=self.bias, scale=self.scale, eye=False)

        self.W_hr = build_linear_pyromodule(in_features = hidden_size, out_features = hidden_size, bias=self.bias, scale=self.scale)
        self.W_hd = build_linear_pyromodule(in_features = hidden_size, out_features = hidden_size, bias=self.bias, scale=self.scale)
        self.W_hn = build_linear_pyromodule(in_features = hidden_size, out_features = hidden_size, bias=self.bias, scale=self.scale)

        self.act = nn.Sigmoid()

    def forward(self, input, h_prev):
        reset_gate = self.act(self.W_ir(input) + self.W_hr(h_prev))
        update_gate = self.act(self.W_id(input) + self.W_hd(h_prev))
        new_gate = nn.Tanh()( self.W_in(input) + reset_gate * ( self.W_hn(h_prev)) )
        hidden_new = (1-update_gate) * new_gate + update_gate * h_prev
        return hidden_new

    @pyro_method
    def hidden2output(self, hidden):
        output = self.W_z(hidden)
        #output = nn.Tanh()(output)
        return output

class ItemHier(PyroModule):
    def __init__(self,
                 item_dim,
                 num_items,
                 item_group=None,
                 hidden_dim=None,
                 device="cpu", **kwargs):
        super().__init__()
        self.item_dim = item_dim
        self.hidden_dim = item_dim if hidden_dim == None else hidden_dim
        self.num_items = num_items
        self.device = device
        self.prior_groupvec_scale = kwargs.get('prior_groupvec_scale', 0.2)
        self.prior_groupscale_scale = kwargs.get('prior_groupscale_scale', 0.2)

        logging.info(f"Initializing itemHier with groupvec ~ N(0, {self.prior_groupvec_scale}), groupscale ~ N(0, {self.prior_groupscale_scale})")

        # Group-vec:
        self.item_group = item_group
        if self.item_group is None:
            self.item_group = torch.zeros((self.num_items, )).long()
        self.num_groups = len(self.item_group.unique())
        self.groupvec = PyroModule[nn.Embedding](num_embeddings=self.num_groups,
                                                 embedding_dim=self.item_dim)
        self.groupvec.weight = PyroSample(
            dist.Normal(
                torch.zeros_like(self.groupvec.weight),
                self.prior_groupvec_scale*torch.ones_like(self.groupvec.weight)
                ).to_event(2))

        self.groupscale = PyroModule[nn.Embedding](
            num_embeddings=self.num_groups, embedding_dim=self.item_dim)

        self.groupscale.weight = PyroSample(
            dist.Normal(
                0.0 * torch.ones_like(self.groupscale.weight), 
                self.prior_groupscale_scale *torch.ones_like(self.groupscale.weight)
                ).to_event(2))

        # Item vec based on group vec hier prior:
        self.itemvec = PyroModule[nn.Embedding](num_embeddings=num_items,
                                                embedding_dim=item_dim,
                                                 padding_idx=0)
        self.itemvec.weight = PyroSample(lambda x: dist.Normal(
            self.groupvec(self.item_group),
            self.groupscale(self.item_group).abs().clamp(0.001)).to_event(2))
    
    def init_set_of_real_parameters(self, true_softmax_mult=1.0, seed=1):
        par_real = {}
        groupvec = generate_random_points_on_d_surface(
            d=self.item_dim, 
            num_points=self.num_groups, 
            radius= 0.1 + 0.7*torch.rand((self.num_groups,1) ),
            max_angle=3.14)
        
        # Scale with softmax mult:
        groupvec = groupvec*true_softmax_mult

        par_real['groupvec.weight'] = groupvec

        par_real['groupscale.weight'] = torch.ones_like(groupvec)*0.1

        # Get item vector placement locally inside the group cluster
        V_loc = generate_random_points_on_d_surface(
            d=self.item_dim, 
            num_points=self.num_items,
            radius=1,
            max_angle=3.14)

        V = (groupvec[self.item_group] + 0.1*true_softmax_mult * V_loc)

        # Special items 0 and 1 is in middle:
        V[:2, :] = 0

        par_real['itemvec.weight'] = V
        return par_real

    @pyro_method
    def forward(self, idx=None):
        if idx is None:
            out = self.itemvec.weight
        else:
            out = self.itemvec(idx)
        return (out) #nn.Tanh()

class ItemFlat(PyroModule):
    def __init__(self,
                 item_dim,
                 num_items,
                 hidden_dim=None,
                 device="cpu", **kwargs):
        super().__init__()
        self.item_dim = item_dim
        self.hidden_dim = item_dim if hidden_dim == None else hidden_dim
        self.num_items = num_items
        self.device = device
        self.prior_groupscale_scale = kwargs.get('prior_groupscale_scale', 0.2)

        logging.info(f"Initializing itemFlat with vec ~ N(0, {self.prior_groupscale_scale})")

        # Item vec based on group vec hier prior:
        self.itemvec = PyroModule[nn.Embedding](num_embeddings=num_items,
                                                embedding_dim=item_dim)
        self.itemvec.weight = PyroSample(lambda x: dist.Normal(
            torch.zeros((num_items, item_dim)),
            self.prior_groupscale_scale*torch.ones((num_items, item_dim))).to_event(2))
    
    def init_set_of_real_parameters(self, seed=1):
        par_real = {}
        return par_real

    @pyro_method
    def forward(self, idx=None):
        if idx is None:
            out = self.itemvec.weight
        else:
            out = self.itemvec(idx)
        return nn.Tanh()(out)

class MeanFieldGuide(PyroModule):
    def __init__(self, model, batch, item_dim, num_users, hidden_dim,freeze_item_parameters=False,freeze_gru_parameters=False, num_samples = 50, h0_amort=None, h0_amort_decayfactor=None,trainer=None, **kwargs):
        super().__init__()
        self.item_dim = item_dim
        self.num_users = num_users
        self.hidden_dim = hidden_dim
        self.maxval = kwargs.get("guide_maxval", 5.0)
        self.maxscale = kwargs.get("guide_maxscale", 0.1)
        self.user_init = kwargs.get("user_init", False)
        self.model = model
        self.dummybatch = batch
        self.freeze_item_parameters = freeze_item_parameters
        self.freeze_gru_parameters = freeze_gru_parameters
        self.num_samples = num_samples
        self.trainer = trainer # report special figures if tb writer is present
        self.initialize_parameters()
        
        ## VI parameters
        self.h0_amort = h0_amort
        if self.h0_amort:
            self.h0_amort_decayfactor = h0_amort_decayfactor
            logging.info(f"Init h0_amort with decayfactor={self.h0_amort_decayfactor}")

    def initialize_parameters(self):
        self.prior_median = {}
        model_trace = poutine.trace(self.model).get_trace(self.dummybatch)

        for node, site in model_trace.iter_stochastic_nodes():
            try:
                samples = site['fn'].sample(sample_shape= (self.num_samples,))
                self.prior_median[node] = samples.median(dim=0)[0]
            except Exception as e:
                self.prior_median[node] = None
                if node in ["data"]:
                    pass
                else:
                    logging.info(f"Could not sample init values for {node}. Exception:")
                    logging.info(e)
    @pyro_method
    def h0_amortization_func(self, batch_click, mask=None, V=None, amort_scalefactor = None):
        if V is None:
            V = pyro.param("model.item_model.itemvec.weight-mean")
        if amort_scalefactor is None:
            amort_scalefactor = pyro.param("h0-amort-scalefactor", torch.tensor(0.2), constraints.interval(0.001,5.0))
        if mask is None:
            mask = torch.ones_like(batch_click)

        if self.freeze_item_parameters:
            V = V.detach()
            amort_scalefactor = amort_scalefactor.detach()


        click_item_ge3 = (torch.gt( batch_click, 3 )*1)
        batch_size, t_maxclick = batch_click.size()
        weight = click_item_ge3/((click_item_ge3.cumsum(dim=1).float()**self.h0_amort_decayfactor)+1e-5)
        weight = weight * mask # Only use those that are in training set
        weight = weight/(weight.sum(dim=1,keepdims=True)+1e-5)

        click_vecs = V[batch_click]
        mean = (click_vecs*weight.unsqueeze(-1)).sum(1)
        # Scale
        weight = click_item_ge3/((click_item_ge3.sum(dim=1, keepdim=True).float())+1e-5)
        V_variation = ((click_vecs - click_vecs.mean(dim=1, keepdim=True))**2)*weight.unsqueeze(-1)
        scale = amort_scalefactor*(V_variation.sum(dim=1) + 1e-6).sqrt()
        return mean, scale


    def __call__(self, batch=None, batch_click=None, temp = 1):
        if batch_click is None:
            batch_click = batch['click']

        posterior = {}

        for node, site in self.prior_median.items():
            par = self.prior_median[node]

            if (node == "user-init-plate") | (node == "data"):
                pass
            elif "user_noise" in node:
                dim = par.size()[1]
                batch_size = len(batch_click)
                mean = torch.zeros((batch_size, dim))
                scale = torch.ones_like(mean)
                posterior[node] = pyro.sample(
                    node,
                    dist.Normal(mean, temp*scale).to_event(1))
            
            elif ('h0-batch' in node ):
                if self.h0_amort:
                    ## HANDLE AMORTIZED INITIAL CONDITION:
                    
                    # Add mask on training data
                    mask=None
                    if batch is not None:
                        if batch.get('mask_type') is not None:
                            mask = (batch['mask_type']==1)
                    
                    mean, scale = self.h0_amortization_func(batch_click, mask=mask)
                    if self.trainer:
                        self.trainer.writer.add_scalar(tag="param/h0-mean-l1",
                                    scalar_value=mean.abs().mean(),
                                    global_step=self.trainer.step)
                        self.trainer.writer.add_scalar(tag="param/h0-scale-l1",
                                    scalar_value=scale.abs().mean(),
                                    global_step=self.trainer.step)
                else:
                    dim = par.size()[1]
                    mean = pyro.param(f"h0-mean",
                                        init_tensor = 0.01*torch.rand((self.num_users, dim)), 
                                        constraint = constraints.interval(-self.maxval,self.maxval))
                    scale = pyro.param(f"h0-scale",
                                        init_tensor= 0.05 +
                                        0*torch.rand((self.num_users, dim)),
                                        constraint=constraints.interval(0,self.maxscale))
                    mean = mean[batch['userId']]
                    scale = scale[batch['userId']]
                #if self.freeze_gru_parameters:
                #    mean = mean.detach()
                #    scale = scale.detach()

                #with pyro.plate("user-init-plate", size = self.num_users, subsample = batch['userId']):
                posterior[node] = pyro.sample(
                    node,
                    dist.Normal(mean, temp*scale).to_event(1))


                    
            elif "user_model.user_scale" in node:
                mean = pyro.param(f"{node}-mean",
                                    init_tensor= 0.01*torch.ones_like(par.detach().clone()), constraint = constraints.interval(0,1.0))
                scale = pyro.param(f"{node}-scale",
                                    init_tensor=0.05 +
                                    0*par.detach().clone(),
                                    constraint=constraints.interval(0,self.maxscale))

                posterior[node] = pyro.sample(
                    node,
                    dist.Normal(mean, temp*scale).independent())
            elif "user_model.gamma" in node:
                mean = pyro.param(f"{node}-mean",
                                    init_tensor= par.detach().clone(), constraint = constraints.interval(0,1.0))
                scale = pyro.param(f"{node}-scale",
                                    init_tensor=0.05 +
                                    0*par.detach().clone(),
                                    constraint=constraints.interval(0,self.maxscale))

                posterior[node] = pyro.sample(
                    node,
                    dist.Normal(mean, temp*scale).independent())
            elif "item_model" in node:
                mean = pyro.param(f"{node}-mean",
                                    init_tensor= 0.01*par.detach().clone())
                scale = pyro.param(f"{node}-scale",
                                    init_tensor= 0.01+ 0*par.detach().clone().abs(),
                                    constraint=constraints.interval(0,self.maxscale))
                if self.freeze_item_parameters:
                    mean = mean.detach()
                    scale = scale.detach()
                posterior[node] = pyro.sample(
                    node,
                    dist.Normal(mean, temp*scale).independent())
            elif "user_model" in node:
                mean = pyro.param(f"{node}-mean",
                                    init_tensor= par.detach().clone(),
                                    constraint = constraints.interval(-self.maxval,self.maxval)
                                    )
                scale = pyro.param(f"{node}-scale",
                                    init_tensor=0.05 +
                                    0*par.detach().clone().abs(),
                                    constraint=constraints.interval(0,self.maxscale))
                if self.freeze_gru_parameters:
                    mean = mean.detach()
                    scale = scale.detach()
                posterior[node] = pyro.sample(
                    node,
                    dist.Normal(mean, temp*scale).independent())
            else:
                mean = pyro.param(f"{node}-mean",
                                    init_tensor= par.detach().clone(),
                                    constraint = constraints.interval(-self.maxval,self.maxval)
                                    )
                scale = pyro.param(f"{node}-scale",
                                    init_tensor=0.05 +
                                    0*par.detach().clone().abs(),
                                    constraint=constraints.interval(0,self.maxscale))
                posterior[node] = pyro.sample(
                    node,
                    dist.Normal(mean, temp*scale).independent())

        return posterior

    def median(self):
        posterior = {}

        for node, site in self.prior_median.items():
            mean = pyro.param(f"{node}-mean")
            posterior[node] = mean

        return posterior

    def get_parameters(self):
        varpar = {}

        for node, par in self.prior_median.items():
            varpar[node] = {}
            varpar[node]['mean'] = pyro.param(f"{node}-mean")
            varpar[node]['scale'] = pyro.param(f"{node}-scale")
        return varpar

#%% MODEL GAMMA WITH POSITIVE POLAR COORD

def generate_random_points_on_d_surface(d, num_points, radius, max_angle=3.14, last_angle_factor=2):

    r = torch.ones((num_points,1))*radius
    angles = torch.rand((num_points, d-1))*max_angle
    angles[:,-1] = angles[:,-1]*last_angle_factor

    cosvec = torch.cos(angles)
    cosvec = torch.cat([cosvec, torch.ones((num_points,1))], dim=1)

    sinvec = torch.sin(angles)
    sinvec = torch.cat([torch.ones((num_points,1)), sinvec], dim=1)
    sincum = sinvec.cumprod(1)

    X = cosvec * sincum*r
    return X


def visualize_3d_scatter(dat):
    import plotly
    import plotly.graph_objs as go
    plotly.offline.init_notebook_mode()

    # Configure the trace.
    trace = go.Scatter3d(
        x=dat[:,0],  # <-- Put your data instead
        y=dat[:,1],  # <-- Put your data instead
        z=dat[:,2],  # <-- Put your data instead
        mode='markers',
        marker={
            'size': 2,
            'opacity': 0.8,
        }
    )

    # Configure the layout.
    layout = go.Layout(
        margin={'l': 0, 'r': 0, 'b': 0, 't': 0}
    )

    data = [trace]

    plot_figure = go.Figure(data=data, layout=layout)

    # Render the plot.
    plotly.offline.iplot(plot_figure)



scorefunctions = {
    'l2' : score_euclidean,
    'dot': score_dot}

user_models = {
    'markov' : UserMarkov,
    'linear' : UserLinear,
    'gru' : UserGRU,
    'lingru' : UserLinearGru
}

item_models = {
    'hier' : ItemHier,
    'flat' : ItemFlat}

user_init_models = {
    "normal": UserInitNormal,
    "compact_normal" : UserInitCompactNormal,
    "zero": UserInitZero
    }