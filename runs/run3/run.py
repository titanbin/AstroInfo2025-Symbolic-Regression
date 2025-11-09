import physo
import numpy as np
import physo.learn.monitoring as monitoring
import pandas as pd
import torch
import os 
import multiprocessing

# Control thread and process settings
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
torch.set_num_threads(1)

# Set the start method for multiprocessing
multiprocessing.set_start_method('fork', force=True)

models=[
'1.5M_A_R4_10',
'500k_A_R2_10',
'500k_A_R4_10',
'500k_C_R4_10',
'250k_A_R2_25',
'250k_A_R2_25_vlk',
'250k_A_R2_10',
'250k_A_R2_5',
'250k_A_R4_25',
'250k_A_R4_25_imf50',
'250k_A_R4_25_lk',
'250k_A_R4_25_retr',
'250k_A_R4_25_vlk',
'250k_A_R4_10',
'250k_A_R4_10_retr',
'250k_B_R4_25',
'250k_B_R4_25_lk',
'250k_C_R2_10',
'250k_C_R4_25',
'250k_C_R4_25_lk',
'250k_C_R4_10',
'250k_W6_R4_25',
'250k_W6_R4_25_retr',
'500k_A_R4_LC_part1',
'500k_A_R4_LC_part2']

def load_model_data_radial(model_name,base_path):
    # Read mass
    rad = pd.read_csv(base_path + model_name + "/radial/0.0-200.0_radial_bin.dat", delimiter=" ", header=None)
    r = rad.to_numpy()  # shape: (N_rows, varying columns)
    
    # Read sigma_phi
    disp_phi = pd.read_csv(base_path + model_name +"/radial/0.0-200.0_disp_phi.dat", delimiter=" ", header=None)
    s_phi = disp_phi.to_numpy()

     # Read sigma_phi
    vel_phi = pd.read_csv(base_path + model_name +"/radial/0.0-200.0_vphi.dat", delimiter=" ", header=None)
    v_phi = vel_phi.to_numpy()
    
    # Read age
    age = pd.read_csv(base_path + model_name + "/age.dat",
                      sep=r"\s+", header=None).to_numpy().flatten()  # shape (N_rows,)

    distance=  int(model_name.split('_')[3])
    
    return {
        "age": age,
        "distance": distance,
        "r": r,
        "disp_phi": s_phi,
        "v_phi": v_phi
    }


def get_data():
    base_path = f"/pbs/throng/training/astroinfo2025/data/Nbody/"
    good_models = [models[i] for i in range(len(models))
               if 'retr' not in models[i] and
               'part' not in models[i] and
               models[i] != '250k_A_R2_5' and
               '1.5M' not in models[i]]

    good_models = [(model,load_model_data_radial(model,base_path)) for model in good_models]

    data = []
    metadata = []
    
    for i,(model_name,model) in enumerate(good_models):
        ages = model['age']
        distance = model['distance']
        closest_age_idx = np.argmin(np.abs(ages-12_000))
        closest_age = ages[closest_age_idx]
        closest_r = model['r'][closest_age_idx]
        closest_v_phi = model['v_phi'][closest_age_idx]
        x = closest_r/np.nanmax(closest_r)
        weights = (np.tanh(10*(x-.45)+1)) + (np.tanh(10*(-x)+1))
        weights -= np.nanmin(weights)
        weights /= np.nanmax(weights)
        weights += .1
        weights /= 1.1
    
        mask = np.logical_and(np.isnan(closest_r) == False,np.isnan(closest_v_phi) == False)
        closest_r = closest_r[mask]
        closest_v_phi = closest_v_phi[mask]
        weights = weights[mask]
        
        metadata.append((model_name,closest_age,distance))
        data.append((closest_r,closest_v_phi,weights))

    X = [np.array(x[0]) for x in data]
    X = [x[np.newaxis,:] for x in X]
    Y = [np.array(x[1]) for x in data]
    weights = [np.array(x[2]) for x in data]

    return X,Y,weights

    
def get_runners():
    save_path_training_curves = 'demo_curves.png'
    save_path_log             = 'demo.log'
    run_logger     = lambda : monitoring.RunLogger(
        save_path = save_path_log,
        do_save = False
    )
    run_visualiser = lambda : monitoring.RunVisualiser (
        epoch_refresh_rate = 1,
        save_path = save_path_training_curves,
        do_show   = False,
        do_prints = True,
        do_save   = True
    )
    
    return run_logger,run_visualiser

def create_config():
    MAX_LENGTH = 35

    # ---------- REWARD CONFIG ----------
    reward_config = {
                     "reward_function"     : physo.physym.reward.SquashedNRMSE,
                     "zero_out_unphysical" : True,
                     "zero_out_duplicates" : False,
                     "keep_lowest_complexity_duplicate" : False,
                     # "parallel_mode" : True,
                     # "n_cpus"        : None,
                    }
    
    # ---------- LEARNING CONFIG ----------
    # Number of trial expressions to try at each epoch
    BATCH_SIZE = int(1e4)
    # Function returning the torch optimizer given a model
    GET_OPTIMIZER = lambda model : torch.optim.Adam(
                                        model.parameters(),
                                        lr=0.0025,
                                                    )
    # Learning config
    learning_config = {
        # Batch related
        'batch_size'       : BATCH_SIZE,
        'max_time_step'    : MAX_LENGTH,
        'n_epochs'         : int(1e9),
        # Loss related
        'gamma_decay'      : 0.7,
        'entropy_weight'   : 0.005,
        # Reward related
        'risk_factor'      : 0.05,
        'rewards_computer' : physo.physym.reward.make_RewardsComputer (**reward_config),
        # Optimizer
        'get_optimizer'    : GET_OPTIMIZER,
        'observe_units'    : True,
    }
    
    # ---------- FREE CONSTANT OPTIMIZATION CONFIG ----------
    free_const_opti_args = {
                'loss'   : "MSE",
                'method' : 'LBFGS',
                'method_args': {
                            'n_steps' : 60,
                            'tol'     : 1e-8,
                            'lbfgs_func_args' : {
                                'max_iter'       : 4,
                                'line_search_fn' : "strong_wolfe",
                                                 },
                                },
            }
    
    # ---------- PRIORS CONFIG ----------
    priors_config  = [
                    #("UniformArityPrior", None),
                    # LENGTH RELATED
                    ("HardLengthPrior"  , {"min_length": 4, "max_length": MAX_LENGTH, }),
                    ("SoftLengthPrior"  , {"length_loc": 8, "scale": 5, }),
                    # RELATIONSHIPS RELATED
                    ("NoUselessInversePrior"  , None),
                    ("PhysicalUnitsPrior", {"prob_eps": np.finfo(np.float32).eps}), # PHYSICALITY
                    ("NestedFunctions", {"functions":["exp",], "max_nesting" : 1}),
                    ("NestedFunctions", {"functions":["log",], "max_nesting" : 1}),
                    #("NestedTrigonometryPrior", {"max_nesting" : 1}),
                    #("OccurrencesPrior", {"targets" : ["1",], "max" : [3,] }),
                     ]
    
    # ---------- RNN CELL CONFIG ----------
    cell_config = {
        "hidden_size" : 128,
        "n_layers"    : 1,
        "is_lobotomized" : False,
    }
    
    # ---------- RUN CONFIG ----------
    config = {
        "learning_config"      : learning_config,
        "reward_config"        : reward_config,
        "free_const_opti_args" : free_const_opti_args,
        "priors_config"        : priors_config,
        "cell_config"          : cell_config,
    }
    return config

def run_sr_analysis(X, y, weights, run_logger=None, run_visualiser=None):
    # Get the number of available CPUs from SLURM environment if available
    n_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
    if n_cpus == 1:
        n_cpus = int(os.environ.get('SLURM_NTASKS', 64))
    
    print(f"Running SR with {n_cpus} CPUs")
    
    # Running SR task
    expression, logs = physo.ClassSR(X, Y, weights,
                                X_units=[[1,0,0]], X_names=["r"], 
                                y_units=[1,-1,0], y_name='v_phi', 
                                class_free_consts_units=[[0,0,0],[0,0,1], [1,0,0], [1,-1,0], [0,1,0]],
                                class_free_consts_names=['C','M', 'r_s', 'v_c', 'T'],
                                fixed_consts_units=[[3,-2,-1]], 
                                fixed_consts=[6.67430*1e-11],
                                spe_free_consts_names = [ "k0"      , "tau"      , "d"  ,"m"],
                                spe_free_consts_units = [ [0, 0, 0] , [0, 1, 0] , [1, 0, 0], [0,0,1] ],
                                op_names=['mul', 'add', 'sub','div', 'inv', 'n2', 'sqrt', 'neg', 'exp', 'log'],
                                run_config = config,
                                epochs=100,
                                parallel_mode=True,
                                n_cpus=n_cpus)

    return expression, logs

if __name__ == '__main__':
    run_logger,run_visualiser = get_runners()
    config = create_config()
    X,Y,weights = get_data()
    run_sr_analysis(X, Y, weights, run_logger=run_logger, run_visualiser=run_visualiser)