from dataclasses import dataclass
from typing import Callable, Tuple, Union
import numpy as np
import torch 
# import optax
import drifts as drifts 
from mean_transport import MarginalFBTM
import argparse
import matplotlib.pyplot as plt 


Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
d = 2
dt = 1e-3
tf = 5
n_time_steps = int(tf / dt)
store_fac = 25


## configure random seed
repeatable_seed = True

if repeatable_seed:
    rng = np.random.default_rng(42)
else:
    rng = np.random.default_rng()



N = 5     
d = 2       
A = 20
r = 0.5     
D = 0.25   
R = np.sqrt(5*N)*r  
B = D/R**2 


D_sqrt = np.sqrt(D)
amp = lambda t: 2
freq = 1
drift = drifts.anharmonic_gaussian

Noisy = True 

circular = True

if bool(circular):
    compute_mut = lambda t: amp(t)*np.array([np.cos(freq*t), np.sin(freq*t)])
else:
    compute_mut = lambda t: amp(t)*np.array([np.cos(freq*t), 0])

## initial distribution parameters
force_args = (A, r, B, N, d, compute_mut)
mu0 = np.tile(compute_mut(0), N)
assert mu0.shape == (N*d,), f"Bad mu0 shape: {mu0.shape}"
sig0 = 0.05

experiment = "Anharmonic"

n_hidden = 4     
n_x_neurons = 4             
n_t_neurons = 128           
n_epochs = 3000


learning_rate = 9e-5
act = torch.nn.GELU          
weight_decay = 5e-5
step = 100
batch_size = 256    

def construct_simulation():

    sim_params = {
    "sig0": sig0,
    "mu0": mu0,
    "drift": drift,
    "force_args": force_args,
    "amp": amp,
    "freq": freq,
    "dt": dt,
    "D": D,
    "D_sqrt": np.sqrt(D),
    "N": N,
    "d": d,
    "Noisy": Noisy, 
    "learning_rate": learning_rate,
    "weight_decay": weight_decay,
    "n_hidden": n_hidden,
    "n_x_neurons": n_x_neurons,
    "n_t_neurons": n_t_neurons,
    "n_epochs": n_epochs,
    "act": act,
    "rng": rng,
    "n_time_steps": n_time_steps,
    "experiment": experiment,
    "batch_size": batch_size, 
    "step": step
    }
    
    tot_params = {** sim_params}
    sim = MarginalFBTM(tot_params)

    return sim



if __name__ == '__main__':
    sim = construct_simulation()
    sim.initialize_network_and_optimizer()
    sim.initialize_forcing()
    clean_trajs, noisy_trajs, ts = sim.generate_training_trajectories()
    # sim.compare_noisy_and_clean(clean_trajs, noisy_trajs)
    learned_vector_field = sim.train_flow_matching(clean_trajs, noisy_trajs)
    initial_conditions = torch.tensor(noisy_trajs[0, :, :-1])

    positions, covs_datapoint_1,cov_comparison_1, t_space, entropy, mean_divergence = sim.mean_flow_trajectory_simulator(initial_conditions, 15, ts, model = learned_vector_field)
    sim.plot_means_and_covs(covs_datapoint_1, cov_comparison_1, t_space)
    sim.plot_trajectories_2d(learned_traj=positions, noisy_traj=noisy_trajs, clean_traj=clean_trajs, n_x_neurons = n_x_neurons, n_t_neurons = n_t_neurons, n_hidden = n_hidden)
    sim.plot_entropy(mean_divergence, entropy, t_space)
