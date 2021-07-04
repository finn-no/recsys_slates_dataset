"""
INSTALLATION FROM PYTORCH IMAGE:
pip install ax-platform
"""
#%%
# Configs deviating from default:
configs = {
    'gru-hier' : {
        'name' : 'gru-hier',
        'user_model_module_first_round': 'linear',
    },
    'lingru-hier' : {
        'name' : 'lingru-hier',
        'user_model_module_first_round': 'linear',
        'user_model_module' : 'lingru',
        'max_freeze_toggles': 1,
    },
    'gru-flat' : {
        'name' : 'gru-flat',
        'user_model_module_first_round': 'linear',
        'remove_item_group': True
    },
    'gru-unicand' : {
        'name' : 'gru-unicand',
        'sample_uniform_action' : True,
        'user_model_module_first_round': 'linear',
        'max_data_evals' : 400
    },
    'linear-hier' : {
        'name' : 'linear-hier',
        'user_model_module_first_round': 'linear',
        'user_model_module' : 'linear',
        'max_freeze_toggles': 0
    },
}


import logging
logging.basicConfig(format='%(asctime)s %(message)s', level='INFO') #removes all training logging from output
from train import *
from ax.plot.contour import plot_contour
from ax.service.ax_client import AxClient
from ax.utils.notebook.plotting import render #, init_notebook_plotting

#%% SET PARAMETERS THAT WE WANT TO OPTIMIZE
parameters=[
    {
    "name": "likelihood_temperature", 
    "type": "range", 
    "bounds": [0.001,1.2], 
    "log_scale": True
    },
    {
    "name": "guide_maxscale", 
    "type": "range", 
    "bounds": [0.05,10.0], 
    "log_scale": False
    },
    {
    "name": "learning_rate", 
    "type": "range", 
    "bounds": [1e-4,1e-2], 
    "log_scale": True
    },
    {
    "name": "prior_rnn_scale", 
    "type": "range", 
    "bounds": [0.1,15.0], 
    "log_scale": False
    },    {
    "name": "prior_groupvec_scale", 
    "type": "range", 
    "bounds": [0.01,15.0], 
    "log_scale": False
    },
    {
    "name": "prior_groupscale_scale", 
    "type": "range", 
    "bounds": [0.01,15.0],
    "log_scale": False
    },
    ]
# %%

list_metrics = ["test/loglik", "valid/loglik", "train/loglik", "train/loss", "valid/loss"] # , "valid/hitrate", "test/hitrate"

def optim_function(parameterization):
    param, ind2val, trainer = main(**parameterization)
    criteria = "valid/loglik"

    # Find best valid_loglik and find all other metrics at that point:
    best_value_criteria = float(max([L[criteria] for L in trainer.epoch_log]).item())
    metrics = [L for L in trainer.epoch_log if L[criteria] == best_value_criteria ][0] # take best epoch for rest of metrics
    metrics = {key: (float(val), 0) for key, val in metrics.items() if (key in list_metrics)}
    for key in ["valid/hitrate", "test/hitrate"]:
        metrics[key] = (trainer.after_training_callback_data[key].item(),0)
    print(metrics)
    return metrics


# quick testfunction
"""import random
import time
def optim_function(parameterization):
    time.sleep(random.randint(5,15))
    score = parameterization['num_particles']
    return {'valid/hitrate': (score,0), 'test/loglik': (score*0.9,0)}
"""
#%% INITIALIZE AX CLIENT:
def initialize(filepath='ax_client_snapshot.json'):
    ax_client = AxClient(verbose_logging=False)
    try:
        ax_client = ax_client.load_from_json_file(filepath=filepath)
    except: 
        logging.warning("COULD NOT LOAD CURRENT EXPERIMENT. STARTING NEW..")
        ax_client.create_experiment(
            name="hypertune_simulation",
            parameters=parameters,
            objective_name="valid/hitrate",
            outcome_constraints=["test/loglik <= 10000"]
        )
    return ax_client

#%% RUN OPTIMIZATION
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run hypertune step for model')
    parser.add_argument('modelname', metavar='modelname',
                        help='model name')
    args = parser.parse_args()
    modelname = args.modelname

    # CONFIG PARS
    config_par = configs[modelname]
    filepath=f"ax_client_{modelname}.json"
    ax_client = initialize(filepath=filepath)

    run_custom_trial=True
    if run_custom_trial:
        filepath=f"ax_client_gru-hier.json"
        ax_client_best = initialize(filepath=filepath)
        custom_par, _ = ax_client_best.get_best_parameters()
        param, trial_index = ax_client.attach_trial(custom_par)

    else:
        param, trial_index = ax_client.get_next_trial()
    logging.info(f"STARTING TRIAL NO.: {trial_index}")
    for key,val in config_par.items():
        param[key] = val

    result = optim_function(param)
    ax_client.complete_trial(trial_index=trial_index, raw_data=result)
    # Checkpoint to keep:
    ax_client.save_to_json_file(filepath=filepath)