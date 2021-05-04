import torch
import matplotlib.pyplot as plt
import pyro
import logging
import os
from torch.utils.tensorboard import SummaryWriter
logging.basicConfig(format='%(asctime)s %(message)s', level='INFO')
from pyro import poutine
import copy
import json
import numpy as np
class PyroTrainer:
    def __init__(self,
                 model,
                 guide,
                 dataloaders,
                 max_epoch=100,
                 before_training_callbacks = [],
                 after_training_callbacks = [],
                 step_callbacks = [],
                 phase_end_callbacks = [],
                 **kwargs):
        self.model = model
        self.guide = guide
        self.dataloaders = dataloaders
        self.max_epoch = max_epoch
        self.step_callbacks = step_callbacks
        self.phase_end_callbacks = phase_end_callbacks
        self.after_training_callbacks = after_training_callbacks
        self.before_training_callbacks = before_training_callbacks
        self.mask2ind = {'train' : 1, 'valid' : 2, 'test' : 3}
        for key, val in kwargs.items():
            setattr(self,key,val)

        self.step = 0  # global step counter (counts datapoints)
        self.epoch = 0

        self.writer = SummaryWriter(
            f'tensorboard/{kwargs.get("name", "default")}')

    def run_before_training_callbacks(self):
        for cb in self.before_training_callbacks:
            l = cb(self)

    def run_after_training_callbacks(self):
        self.after_training_callback_data = {}
        for cb in self.after_training_callbacks:
            try:
                l = cb(self)
                for key, val in l.items():
                    self.after_training_callback_data[key] = val
            except Exception as e:
                logging.info(f"Could not run callback. exception:")
                print(e)

        self.writer.flush() #flush all logging to disk before we stop

    def run_step_callbacks(self, phase, batch):
        batch = {key: val.long().to(self.device) for key, val in batch.items()}
        # Special masking operation (move to dataloader?)
       
        batch['phase_mask'] = (batch['mask_type']==self.mask2ind[phase]).float()


        # EXECUTES ALL CALLBACKS RELATED TO EACH STEP BATCH
        tmp_log = {}
        tmp_log['num_obs'] = ((batch['click']!=0)*batch['phase_mask']).sum()
        for cb in self.step_callbacks:
            l = cb(self, phase=phase, batch=batch)

            for key, val in l.items():
                tmp_log[key] = val

        return tmp_log

    def run_phase_end_callbacks(self, phase, logs):
        for cb in self.phase_end_callbacks:
            l = cb(self, phase=phase, logs=logs)


    def fit(self):
        # Initialize an epoch log
        self.epoch_log = list()
        self.run_before_training_callbacks()

        while self.epoch <= self.max_epoch:
            self.epoch_log.append({'epoch': self.epoch, 'stop': False})
            logging.info("")
            logging.info('-' * 10)
            logging.info(f'Epoch {self.epoch}/{self.max_epoch} \t Step {self.step}')

            for phase, dl in self.dataloaders.items():
                logs = []
                batch_size = dl.batch_size  # assume that batch size is constant (small error in last step)

                for batch in dl:
                    tmp_log = self.run_step_callbacks(phase, batch)
                    
                    if phase == "train":
                        self.step += batch_size
                        self.writer.add_scalar("epoch", self.epoch, global_step=self.step)

                    # Add tmp log to log list for phase
                    logs.append(tmp_log)

                self.run_phase_end_callbacks(phase=phase, logs=logs)

            
            if self.epoch_log[-1]['stop']:
                logging.info('Training stopped: A callback wanted early stopping.')
                break
            self.epoch += 1

        self.run_after_training_callbacks()

#######
## CALLBACKS
#######



class VisualizeEmbeddings:
    def __init__(self, sim=None, ind2val = None):
        self.sim = sim
        self.ind2val = ind2val

    def __call__(self, trainer, **kwargs):
        # Visualize item vectors:
        # %% PLOT OF H0 parameters of users
        if trainer.model.user_init_module!="zero":
            try:
                num_plot_users = 1000
                h0 = pyro.param("h0-mean").detach().cpu()[:num_plot_users]
                
                if self.sim:
                    usergroup=self.sim.env.user_init_group[:num_plot_users].cpu()
                else:
                    usergroup=None

                trainer.writer.add_embedding(tag="h0",mat= h0, metadata=usergroup, global_step=trainer.step)
            except:
                logging.info("could not visualize h0")
        try:
            # %% PLOT OF item vector parameters
            V = pyro.param('model.item_model.itemvec.weight-mean').detach().cpu()
            if self.ind2val:
                labels = [self.ind2val['category'][int(i)] for i in trainer.model.item_group.cpu()]
                group_labels = self.ind2val['category'].values()
            else:
                labels = [int(i) for i in trainer.model.item_group.cpu()]
                group_labels=None

            idx = torch.randint(V.size(0), size=(10000,))
            trainer.writer.add_embedding(tag="item_vectors",mat= V[idx], metadata=[labels[i] for i in idx], global_step=trainer.step+1)

            #%% PLOT ITEM GROUPS
            Vg = pyro.param("model.item_model.groupvec.weight-mean").detach().cpu().numpy()
            trainer.writer.add_embedding(tag="group_vectors",mat= Vg, metadata=group_labels, global_step=trainer.step)
        except:
            logging.info("Could not visualize item embeddings")
            pass

class RewardComputation:
    def __init__(self, param, test_sim):
        self.param = param
        self.calc_footrule=False
        self.test_sim = test_sim

    def __call__(self, trainer):
        ### SIMULATE REWARDS..:
        rec_types = {
            'thompson' : trainer.model.recommend,
            'inslate' : trainer.model.recommend_inslate
            }

        for rectype, recommend_func in rec_types.items():
            logging.info(f"compute {rectype} reward..")
            t_start = self.param.get("t_testsplit")
            current_data = copy.deepcopy(trainer.dataloaders['train'].dataset.data)
            
            # zero pad the period we want to test:
            for key, val in current_data.items():
                if key not in ["userId", "mask_type"]:
                    current_data[key][:,t_start:] = 0

            self.test_sim.reset_data(data=current_data)
            all_rewards = self.test_sim.play_game(
                recommend_func,
                par=lambda batch: trainer.guide(batch, temp=1.0),
                userIds = current_data['userId'],
                t_start=t_start
                )

            train_mask = (current_data['mask_type'][:,t_start:]==1).float()
            valid_mask = (current_data['mask_type'][:,t_start:]==2).float()
            test_mask =  (current_data['mask_type'][:,t_start:]==3).float()

            reward_train_timestep = (all_rewards * train_mask).sum(0) / train_mask.sum(0)
            reward_valid_timestep = (all_rewards * valid_mask).sum(0) / valid_mask.sum(0)
            reward_test_timestep = (all_rewards * valid_mask).sum(0) / valid_mask.sum(0)
            trainer.epoch_log[-1][f'train/reward-{rectype}'] = reward_train_timestep.mean()

            trainer.epoch_log[-1][f'valid/reward-{rectype}'] = reward_valid_timestep.mean()
            trainer.epoch_log[-1][f'test/reward-{rectype}'] = reward_test_timestep.mean()
            # log per timestep:
            for i in range(all_rewards.size()[1]):
                trainer.writer.add_scalar(f"reward_time/train-{rectype}", reward_train_timestep[i], global_step=i+t_start)
            for i in range(all_rewards.size()[1]):
                trainer.writer.add_scalar(f"reward_time/valid-{rectype}", reward_valid_timestep[i], global_step=i+t_start)
            for i in range(all_rewards.size()[1]):
                trainer.writer.add_scalar(f"reward_time/test-{rectype}", reward_test_timestep[i], global_step=i+t_start)

            u = 1
            anipath = AnimatePath(sim = self.test_sim, model=trainer.model, guide = trainer.guide, num_samples=100)
            for t in [0, 5, 10, 15, 19]:
                anipath.step(t=t, u = u)
                trainer.writer.add_figure(f"visualize-path-{rectype}", plt.gcf(), global_step=t)

        ### CALC FOOTRULE DISTANCE BETWEEN REAL AND ESTIMATED RECS:
        # calc for all data at last timestep:
        if self.calc_footrule:
            logging.info("Compute FOOTRULE distance..")
            num_items = self.param['num_items']-3
            argsort_real = trainer.model.recommend(trainer.dataloaders['train'].dataset.data, par="real", num_rec = num_items)
            argsort_estimated = trainer.model.recommend(trainer.dataloaders['train'].dataset.data, par=trainer.guide, num_rec = num_items)
            rank_real = argsort2rank_matrix(argsort_real.long(), num_items = num_items+3)
            rank_estimated = argsort2rank_matrix(argsort_estimated.long(), num_items = num_items+3)
            
            train_idx = train_mask[:,-1]
            trainer.epoch_log[-1][f'train/footrule'] = dist_footrule(rank_real[train_idx.bool()], rank_estimated[train_idx.bool()])
            trainer.epoch_log[-1][f'valid/footrule'] = dist_footrule(rank_real[(1-train_idx).bool()], rank_estimated[(1-train_idx).bool()])

class ReportHparam:
    def __init__(self, hyperparam):
        self.hyperparam = copy.deepcopy(hyperparam)

    def __call__(self, trainer, **kwargs):
        ### Add hyperparameters and final metrics to hparam:
        serialized_param = json.loads(json.dumps(self.hyperparam, default=str))
        trainer.writer.add_hparams(hparam_dict=serialized_param,
                                metric_dict=trainer.epoch_log[-1])



@torch.no_grad()
def checksum_data(trainer, **kwargs):
    checksum = sum([val.float().mean() for key, val in trainer.dataloaders['train'].dataset.data.items()])
    trainer.writer.add_scalar("data_checksum", checksum, global_step=0)

class ReportPyroParameters:
    def __init__(self, report_param_histogram=False):
        self.report_param_histogram = report_param_histogram

    def __call__(self, trainer, phase, **kwargs):
        if phase == "train":
            # Report all parameters
            try:
                gamma = pyro.param("gamma-mean")
                if len(gamma)>1:
                    for i in range(len(gamma)):
                        trainer.writer.add_scalar(tag=f"param/gamma_{self.ind2val['displayType'][i]}", scalar_value = gamma[i], global_step=self.step)
            except:
                pass
            for name, par in pyro.get_param_store().items():
                trainer.writer.add_scalar(tag=f"param/{name}-l1",
                                        scalar_value=par.abs().mean(),
                                        global_step=trainer.step)
                if self.report_param_histogram:
                    trainer.writer.add_histogram(tag=f"param/{name}",
                                                values=par,
                                                global_step=trainer.step)

class Simulator_batch_stats:
    def __init__(self, sim, **kwargs):
        self.sim = sim

    @torch.no_grad()
    def __call__(self, trainer, phase, batch):
        stats = {}
        res_hat = pyro.condition(lambda batch: trainer.model(batch),
                                data=trainer.guide(batch, temp=0.0001))(batch)


        res = self.sim.env.likelihood(batch)

        # Compute probabilities
        score2prob = lambda s: s.exp() / (s.exp().sum(2, keepdims=True))
        res['prob'] = score2prob(res['score'])
        res_hat['prob_hat'] = score2prob(res_hat['score'])

        prob_mae_unmasked = (res['prob'] - res_hat['prob_hat'])
        
        time_mask = (batch['click'] != 0)*(batch['click'] != 2).float()
        mask = time_mask*batch['phase_mask']
        masked_prob = mask.unsqueeze(2)*prob_mae_unmasked
        stats['prob-mae'] = masked_prob.abs().sum() / mask.sum()
        return stats

@torch.no_grad()
class CalcBatchStats:
    def __init__(self, likelihood_temperature, prior_temperature, **kwargs):
        self.likelihood_temperature = likelihood_temperature
        self.prior_temperature = prior_temperature
        
    @torch.no_grad()
    def __call__(self, trainer, phase, batch):
        stats = {}
        
        ## LIKELIHOODS
        guide_mode = lambda *args, **kwargs: trainer.guide(temp=0.0001, *args, **kwargs)
        guide_trace = poutine.trace(guide_mode).get_trace(batch)
        model_with_guidepar = poutine.replay(trainer.model, trace=guide_trace)
        model_trace = poutine.trace(model_with_guidepar).get_trace(batch)
        model_trace.compute_log_prob()
        guide_trace.compute_log_prob()

        logguide = guide_trace.log_prob_sum().item()
        # raw value calc:
        raw_loglik = model_trace.nodes['obs']['log_prob'].sum()

        # compute elbo:
        totlogprob = model_trace.log_prob_sum().item()
        logprior = totlogprob - raw_loglik

        # Report values that are rescaled:
        stats['loglik'] = raw_loglik*self.likelihood_temperature
        #stats['loglik_neg'] = -stats['loglik']
        

        stats['KL_pq'] = logguide - logprior*self.prior_temperature
        stats['elbo'] = stats['loglik'] - stats['KL_pq']
        return stats


# 
@torch.no_grad()
def report_phase_end(trainer, phase, logs, **kwargs):
    """ Function that reports all values (scalars) in logs to trainer.writer """
    keys = logs[0].keys()  # take elements of first dict and they are all equal
    summed_stats = {key: sum([l[key] for l in logs]) for key in keys}

    num = len(logs)
    # Add summed stats to epoch_log:
    for key, val in summed_stats.items():
        trainer.epoch_log[-1][f"{phase}/{key}"] = val / num

    # Report epoch log to tensorboard:
    for key, val in trainer.epoch_log[-1].items():
        if phase in key: # only report if correct phase
            trainer.writer.add_scalar(tag=key,
                                    scalar_value=val,
                                    global_step=trainer.step)
    
    logging.info(f"phase: {phase} \t loglik: {trainer.epoch_log[-1][f'{phase}/loglik']:.1f} \t loss: {trainer.epoch_log[-1][f'{phase}/loss']:.1f}")


class SviStep:
    def __init__(
        self, 
        model, 
        guide,
        learning_rate=1e-2,
        beta1 = 0.9,
        beta2 = 0.999,
        eps = 1e-8,
        clip_norm=10,
        num_particles = 1,
        device = "cpu",
        **kwargs
        ):
        self.model = model
        self.guide = guide
        self.clip_norm = clip_norm
        self.learning_rate = learning_rate
        self.num_particles = num_particles
        self.device = device
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.init_opt()

    def init_opt(self):
        logging.info(f"Initializing default Adam optimizer with lr={self.learning_rate}, clip_norm={self.clip_norm}, num_particles ={self.num_particles}")
        self.svi = pyro.infer.SVI(model=self.model,
                                  guide=self.guide,
                                  optim=pyro.optim.Adam({
                                      "lr": self.learning_rate, "betas": (self.beta1, self.beta2), 'eps' :self.eps},{ 
                                      "clip_norm" : self.clip_norm}),
                                  loss=pyro.infer.Trace_ELBO(num_particles=self.num_particles))
        return True

    def __call__(self, trainer, phase, batch):
        stats = {}
        if phase == "train":
            stats['loss'] = self.svi.step(batch)
        else:
            stats['loss'] = self.svi.evaluate_loss(batch)
        
        return stats

class EarlyStoppingAndCheckpoint:
    def __init__(
        self, 
        stopping_criteria,
        patience=1, 
        save_dir="checkpoints", 
        name = "parameters", 
        **kwargs):
        self.stopping_criteria = stopping_criteria
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.best_loss = None
        self.patience = patience
        self.counter = 0
        self.path = f"{self.save_dir}/{name}.pyro"

    def __call__(self, trainer, phase, logs, **kwargs):
        if phase != "train":
            loss = trainer.epoch_log[-1][self.stopping_criteria]
            if (self.best_loss is None):
                self.best_loss = loss
                self.save_checkpoint()
                self.counter = 0
                return False
            elif loss < self.best_loss:
                self.best_loss = loss
                self.save_checkpoint()
                self.counter = 0
                return False
                
            elif loss >= self.best_loss:
                self.counter += 1
                if self.counter > self.patience:
                    logging.info(f"REACHED EARLY STOPPING ON EPOCH {trainer.epoch}")
                    self.load_checkpoint()
                    trainer.epoch_log[-1]['stop'] = True
                    return True

    def save_checkpoint(self):
        logging.info(f"Saving model to {self.path}..")
        pyro.get_param_store().save(self.path)

    def load_checkpoint(self):
        logging.info(f"Loading latest checkpoint from {self.path}.. (+ clear param store first)")
        pyro.clear_param_store()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()
        pyro.get_param_store().load(self.path)

class AlternateTrainingScheduleCheckpoint:
    def __init__(
        self, 
        model, 
        guide,
        stopping_criteria, 
        patience_toggle, 
        user_model_module, 
        user_model_module_first_round=None,
        max_freeze_toggles=1,
        svi_step_callback=None, 
        save_dir="checkpoints", 
        name = "parameters", 
        max_epochs=100,
        force_first_toggle_epoch=None,
        *args, **kwargs):

        self.model = model
        self.guide = guide
        self.user_model_module = user_model_module
        self.user_model_module_first_round = user_model_module_first_round
        self.svi_step_callback = svi_step_callback
        
        self.counter = 0
        self.best_loss = None
        self.stopping_criteria =stopping_criteria
        self.patience_toggle = patience_toggle
        self.max_epochs=max_epochs
        if force_first_toggle_epoch is None:
            force_first_toggle_epoch = int(max_epochs/2)
        self.force_first_toggle_epoch=force_first_toggle_epoch
        self.max_freeze_toggles = max_freeze_toggles
        self.toggle_count = 0
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.path = f"{self.save_dir}/{name}.pyro"
        logging.info(f"Initializing item/user freeze toggling with criteria={self.stopping_criteria}")

        # Set model to be markov first round:
        if self.user_model_module_first_round is not None:
            self.set_user_model(self.user_model_module_first_round)
            logging.info(f"Overwriting user module to be {self.user_model_module_first_round} for first Freeze round.")

    def __call__(self, trainer, phase, logs, *args, **kwargs):
        if phase == "valid":
            loss = trainer.epoch_log[-1][self.stopping_criteria]
            
            if (
                (trainer.epoch_log[-1]['epoch'] > self.force_first_toggle_epoch)
                & (self.toggle_count==0)
                & (self.best_loss is not None)
                ):
                logging.info(f"Toggle has not happened before epoch {self.force_first_toggle_epoch}. Forcing toggle..")
                loss = self.best_loss+1
                self.counter = self.patience_toggle+1
            if (self.best_loss is None):
                self.best_loss = loss
                self.counter = 0
                self.save_checkpoint()

            elif loss < self.best_loss:
                self.best_loss = loss
                self.counter = 0
                self.save_checkpoint()
                
            elif loss >= self.best_loss:
                self.counter += 1
                if self.counter > self.patience_toggle:
                    logging.info(f"Reached toggle patience of freeze-iterator.")
                    if self.toggle_count <self.max_freeze_toggles:
                        self.toggle()
                        self.counter = 0
                        self.toggle_count +=1
                    else:
                        logging.info("Max toggles reached. Early stopping")
                        trainer.epoch_log[-1]['stop'] = True
                    self.load_checkpoint()

            trainer.writer.add_scalar(tag=f"param/freeze_item_parameters", scalar_value = float(self.guide.freeze_item_parameters), global_step=trainer.step)
            trainer.writer.add_scalar(tag=f"param/freeze_gru_parameters", scalar_value = float(self.guide.freeze_gru_parameters), global_step=trainer.step)
            return {}
    
    def set_user_model(self, user_module):
            import bayesrec.models as models
            # save and load checkpoints to avoid losing them when overwriting user_model..
            self.save_checkpoint()
            self.model.param['user_model_module'] = user_module
            self.model.user_model = models.user_models[self.model.param.get('user_model_module')](**self.model.param)
            self.load_checkpoint()
            self.guide.initialize_parameters()

    def toggle(self):
        logging.info("Load previous best checkpoint before toggling..")
        self.load_checkpoint()
        # Reinit user model to be gru:
        if self.model.param['user_model_module'] != self.user_model_module:
            self.set_user_model(self.user_model_module)

        self.guide.freeze_item_parameters = bool(True-self.guide.freeze_item_parameters)
        self.guide.freeze_gru_parameters = bool(True-self.guide.freeze_gru_parameters)
        self.best_loss = None
        logging.info(f"Toggled user/item-paramters. Freezed item/user {self.guide.freeze_item_parameters}/{self.guide.freeze_gru_parameters}")
        if self.svi_step_callback:
            logging.info("Resetting svi step optimizer to zero out gradients")
            self.svi_step_callback.init_opt()
        
    def save_checkpoint(self):
        logging.info(f"Saving model to {self.path}..")
        pyro.get_param_store().save(self.path)

    def load_checkpoint(self):
        logging.info(f"Loading latest checkpoint from {self.path}.. (+ clearing param store first)")
        pyro.clear_param_store()
        pyro.get_param_store().load(self.path)

#%%

def plotM(M, type = "scatter", **kwargs):
    import seaborn as sns
    if type == "line":
        p = sns.lineplot
    else:
        p = sns.scatterplot
    return p(M[:,0], M[:,1], **kwargs)


class AnimatePath:
    def __init__(self, sim, model, guide, num_samples=1):
        self.sim = sim
        self.num_samples = num_samples
        self.sim.data['phase_mask'] = torch.ones_like(sim.data['click'])
        self.pars = []
        for i in range(self.num_samples):
            temp = 1e-6 if i==0 else 1.0
            output = model.likelihood(self.sim.data, par=guide(self.sim.data, temp=temp))
            output = {key : val.detach().cpu() for key, val in output.items()}
            self.pars.append(output)

    def add_visualization_to_tensorboard(self, trainer, step, u=1):
        self.step(t=step, u = u)
        plt.xlim((-0.2,0.2))
        plt.ylim((-0.2,0.2))
        trainer.writer.add_figure(f"visualize-path", plt.gcf(), global_step=step)

    def step(self, t, u):

        # for each time step:
        action = self.sim.data['action'][u,t].cpu()
        click = self.sim.data['click'][u,t].unsqueeze(0).cpu()

        # Plot all items:
        p = plotM(self.pars[0]['V'], alpha = 0.1)

        
        # Plot all recommended actions:
        #plotM(self.pars[0]['V'][action], color =['yellow'], alpha = 0.5)
        #print(action.size())
        for i in range(self.num_samples):
            plotM(self.pars[i]['V'][action], color =['gold'], alpha = 1.0 if i==0 else 0.5)

            
            plotM(self.pars[i]['V'][click], color = ['red'], alpha=0.9)
            
            zt = self.pars[i]['zt'][u,t].unsqueeze(0).cpu()
            plotM(zt, color = ['blue'], alpha=0.5)

        # Plot corresponding click:
        
        plt.legend(labels=['all items', 'recommended items', 'click item', 'zt (estimated)'])
        
        text = f"t={t}"
        if all(click==1):
            text += ": noClick"
        plt.title(label = text)
### FOOTRULE FUNCTIONS
def argsort2rank(idx, num_items=None):
    # create rank vector from the indicies that returns from torch.argsort()
    if num_items is None:
        num_items = len(idx)
    rank = torch.zeros((num_items,)).long()
    rank[idx] = torch.arange(0,len(idx))
    return rank

def argsort2rank_matrix(idx_matrix, num_items=None):
    batch_size, nc = idx_matrix.size()
    if num_items is None:
        num_items = nc

    rank = torch.zeros((batch_size, num_items)).long()

    for i in range(batch_size):
        rank[i,:] = argsort2rank(idx_matrix[i,:], num_items=num_items)
    return rank

def dist_footrule(r1, r2):
    return (r1-r2).abs().float().mean()

def calc_hitrate(trainer, phase = "train", **kwargs):
    try:
        K = 20
        t_testsplit = trainer.t_testsplit

        #%% HITRATE
        all_data = trainer.dataloaders['train'].dataset.data
        #choose 100k first users (they are already randomized elsewhere):
        all_data = {key: val[:100000] for key, val in all_data.items()} 
        all_data = {key: val.long().to(trainer.device) for key, val in all_data.items()}
        guide_MAP = lambda *args, **kwargs: trainer.guide(temp=0, *args, **kwargs)
        recs = trainer.model.recommend(all_data, par=guide_MAP, num_rec=K, t_rec=t_testsplit, chunksize=15)
        # %%
        recs = recs.detach().cpu()
        clicks = all_data['click'][:,(t_testsplit+1):].detach().cpu()

        dataset_type = all_data['mask_type'][:,6]
        hit_vec = torch.zeros_like(dataset_type)
        for i in range(len(recs)):
            intersect = np.intersect1d(clicks[i], recs[i])
            hits = len(intersect)
            if hits>0:
                hit_vec[i] = hits
        # %%
        ind2mask = {ind:name for name, ind in trainer.mask2ind.items() }
        hitrates = {}
        for idx, phase in ind2mask.items():
            hitrate = hit_vec[dataset_type==idx].float().mean()
            trainer.writer.add_scalar(tag=f"{phase}/hitrate", scalar_value = hitrate, global_step=trainer.step)
            logging.info(f"{phase}/hitrate : {hitrate:.2g}")
            hitrates[f"{phase}/hitrate"] = hitrate
        return hitrates
    except Exception as e:
        logging.warning("Could not compute hitrate. Exception:")
        print(e)