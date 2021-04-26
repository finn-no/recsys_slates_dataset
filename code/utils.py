import yaml
import names
import logging
import copy
import os
logging.basicConfig(format='%(asctime)s %(message)s', level='INFO')


def load_param(**kwargs):
    """ 
    Load params has three levels of inserting with decreasing priority:
        1. **kwargs when calling this function has priority
        2. config.yml in working directory
        3. If no parameter is supplied, a default parameter from config_default.yml is used.
    """
    # Load default config:
    this_dir, this_filename = os.path.split(__file__)
    param = yaml.safe_load(open(f"{this_dir}/config_default.yml", 'r'))

    # Overwrite params with fields in config.yml:
    config_param = yaml.safe_load(open("config.yml", 'r'))
    for key, val in config_param.items():
        logging.info(f"Overwriting parameter {key} to {val}.")
        param[key] = val
    # Overwrite param with whatever is in kwargs:
    
    try:
        for key, val in kwargs.items():
            logging.info(f"Overwriting parameter {key} to {val}.")
            param[key] = val
    except:
        logging.info("ERROR: Did no overwrite of default param.")

    # generate random name
    if len(kwargs)>0:
        try:
            name = ":".join([f"{key[:3]}={val}" for key, val in kwargs.items()])
            name = param.get('name') +":"+ name[:100] # reduce length of name
        except:
            name = "kwargs-"
    else:
        name=param.get('name')
    
    random_name = f"{names.get_full_name().replace(' ','-')}"
    param['name'] = f"{random_name}-{name}"[:100]

    logging.info("--------")
    logging.info(f"--- LOADED MODEL {param['name']}")
    logging.info("--------")

    if param.get("max_data_evals"):
        logging.info("max_data_evals given. Adjusting max epoch and patience..")
        param['max_epochs'] = int(param['max_data_evals']/param['num_particles'])
        if param.get('max_freeze_toggles') == 0: # double if no toggles
            param['max_epochs'] = param['max_epochs']*2
            logging.info("Doubling epochs as one step opt")
        param['patience_toggle'] = int(param['max_epochs']*0.1)
    return param

def load_sim_param(**kwargs):
    logging.info(f"--- LOADING SIMULATION PARS")
    param = load_param()
    # Build simulation parameters (copy inn default config if it doesnt exist in sim)
    simconfig = yaml.safe_load(open("config_simulation.yml","r"))
    # overwrite with simconfig:
    for key, val in simconfig.items():
        param[key] = val
    return param

import random

def get_best_cuda():
    import gpustat
    stats = gpustat.GPUStatCollection.new_query()
    ids = map(lambda gpu: int(gpu.entry['index']), stats)
    ratios = map(lambda gpu: float(gpu.entry['memory.used'])/float(gpu.entry['memory.total']), stats)
    pairs = list(zip(ids, ratios))
    random.shuffle(pairs)
    bestGPU = min(pairs, key=lambda x: x[1])[0]
    return f"cuda:{bestGPU}"