from dataclasses import dataclass
from typing import Callable, Tuple, Union
import numpy as np
import torch 
from data import generate_training_data
from model import construct_mean_flow_model
from training import MeanFlowMatchingTrainer
State = np.ndarray
Time = float
from datetime import datetime 
import os 
from matplotlib.lines import Line2D
from functorch import jacrev

import matplotlib.pyplot as plt

Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def Cd(t):
    return 0.490909 * torch.exp(-11 * t / 10) * (-0.159722 - 0.712963 * torch.exp(t / 10) + torch.exp(11 * t / 10))

def Co(t):
    return 0.00909091 * torch.exp(-11 * t / 10) * (-8.625 + 9.625 * torch.exp(t / 10) - 1.0 * torch.exp(11 * t / 10))

def evaluate_analytical_cov(t, N=50, device=Device):
    cd = Cd(t).to(device)
    co = Co(t).to(device)  # Not needed for trace, but useful for full matrix
    
    # This matches your Mathematica exactly
    trace = N * cd
    
    # If you need the full matrix for entropy:
    I_N = torch.eye(N, device=device)
    J_N = torch.ones((N, N), device=device)
    C_t = (cd - co) * I_N + co * J_N
    
    entropy = N * (torch.log(2*torch.pi*torch.tensor(1.0)) + 1) + 0.5 * torch.logdet(C_t)
   
    return trace.to(device), entropy.to(device)

def evaluate_sde_cov(n, n_steps, trajs): 
    step = int(5000 / n_steps)
    noisy_traj_pos = torch.tensor(trajs[int(step * n), :, :-1])
    cov_par = torch.trace(torch.cov(noisy_traj_pos))
    print(f'inter particle covariance: {cov_par}')
    cov_dim = torch.trace(torch.cov(noisy_traj_pos.T))
    print(f'inter dimension covariance: {cov_dim}')
    return cov_par * cov_dim
    


def compute_analytical_entropy_production_rate(analytical_entropy): 
    print(analytical_entropy.size())
    dt = 1/(analytical_entropy.size(0)-1)
    entropy_prod = torch.zeros(size=(analytical_entropy.size(0)-1,))
    for i in range(analytical_entropy.size(0)-1): 
        entropy_prod[i] = (analytical_entropy[i+1] - analytical_entropy[i] ) / dt

    return entropy_prod 

class MarginalFBTM():

    def __init__(self, data_dict: dict) -> None:
        self.__dict__ = data_dict.copy()
    
    sig0: float
    mu0: np.ndarray

    # system parameters
    drift: Callable[[State, Time], State]
    force_args: Tuple
    amp: Callable[[Time], float]
    freq: float
    dt: float
    D: np.ndarray
    D_sqrt: np.ndarray
    d: int
    N: int
    weight_decay: float
    learning_rate: float

    rng: np.random.Generator

    n_time_steps: int
    batch_size: int
    step: int

    def initialize_forcing(self) -> None:
        self.forcing = lambda x, t: self.drift(x, t, *self.force_args)


    def initialize_network_and_optimizer(self) -> None:
        
        self.network = construct_mean_flow_model(input_dim=self.d, output_dim=self.d, dim=self.n_x_neurons, n_hidden=self.n_hidden, act= self.act, time_embed_dim=self.n_t_neurons)
        self.opt = torch.optim.RAdam(
                    self.network.parameters(), 
                    self.learning_rate, 
                    weight_decay=self.weight_decay
                )
        

    def generate_training_trajectories(self):
        print('Generating Trajectories ...')
        x0 = self.mu0.reshape(self.N, self.d) + self.sig0 * self.rng.normal(size=(self.N, self.d))
        self.clean_trajs, self.noisy_trajs, ts = generate_training_data(self.n_time_steps, self.dt, self.N, self.d, 
                            self.D_sqrt, x0=x0, forcing=self.forcing, rng=self.rng, Noisy = self.Noisy)
        
        return self.clean_trajs, self.noisy_trajs, ts



    def plot_entropy(self, model_entropy, analytical_entropy, timepoints):
        os.makedirs('results', exist_ok=True)
        plt.figure(figsize=(12, 6))
        # Convert tensors to numpy arrays for plotting
        model_entropy = [item.cpu().detach().numpy() if isinstance(item, torch.Tensor) else item 
                        for item in model_entropy]

        timepoints = [item.cpu().detach().numpy() if isinstance(item, torch.Tensor) else item 
                        for item in timepoints]

        
        # Plot with actual epoch numbers on x-axis
        plt.plot(timepoints[:-1], model_entropy[:-1], 'o-', markersize=2)
        if self.experiment == "Harmonic":
            entropy_prod = compute_analytical_entropy_production_rate(analytical_entropy)
            plt.plot(timepoints[:-1], entropy_prod, 'x-', markersize=2)
        plt.xlabel('Timepoint', fontsize=16)
        plt.ylabel('Entropy Production Rate', fontsize=16)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.experiment}_entropy_ts{timestamp}.png"
        title = f"Entropy Production Rate for { self.experiment } Experiment"
        plt.title(title, fontsize=18)
        plt.grid(True)
        plt.savefig(f'results/{filename}')
        plt.show()


    def plot_trajectories_2d(self, learned_traj, noisy_traj, clean_traj, n_x_neurons, n_t_neurons, n_hidden):
        """Plot three trajectory types side-by-side with fixed y-axis limits"""
        # Convert tensors to numpy if needed
        learned_traj = learned_traj.cpu().detach().numpy()
        noisy_traj = noisy_traj.cpu().detach().numpy() if isinstance(noisy_traj, torch.Tensor) else noisy_traj
        clean_traj = clean_traj.cpu().detach().numpy() if isinstance(clean_traj, torch.Tensor) else clean_traj

        os.makedirs('results', exist_ok=True)
        tp, par, dim = learned_traj.shape
        
        # Create figure with three subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 6))
        
        # Style parameters
        colors = plt.cm.tab10(np.linspace(0, 1, par))
        marker_size = 60
        line_width = 1.5
        alpha = 0.8
        
        # Set consistent y-axis limits for all plots
        if self.experiment == "Harmonic": 
            y_limits = (-2.5, 1.5)
            x_limits = (-2, 3)
        if self.experiment == "Anharmonic": 
            y_limits = (-1, 4.0)
            x_limits = (0, 4)

        # Plot 1: Noisy Trajectories (Input Data)
        for p in range(par):
            ax1.plot(noisy_traj[:, p, 0], noisy_traj[:, p, 1],
                    color=colors[p], linestyle='-', linewidth=line_width, alpha=alpha)
            ax1.scatter(noisy_traj[0, p, 0], noisy_traj[0, p, 1],
                    color=colors[p], marker='o', s=marker_size, 
                    edgecolor='black', zorder=3)
            ax1.scatter(noisy_traj[-1, p, 0], noisy_traj[-1, p, 1],
                    color=colors[p], marker='X', s=marker_size,
                    edgecolor='black', zorder=3)
        ax1.set_title('Noisy Trajectories', fontsize=16)
        ax1.set_xlabel("X Position", fontsize=14)
        ax1.set_ylabel("Y Position", fontsize=14)
        # ax1.set_xlim(x_limits)  # Set y-axis limits
        # ax1.set_ylim(y_limits)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Learned Trajectories (Model Output)
        for p in range(par):
            ax2.plot(learned_traj[:, p, 0], learned_traj[:, p, 1],
                    color=colors[p], linestyle='--', linewidth=line_width, alpha=alpha)
            ax2.scatter(learned_traj[0, p, 0], learned_traj[0, p, 1],
                    color=colors[p], marker='o', s=marker_size,
                    edgecolor='black', zorder=3)
            ax2.scatter(learned_traj[-1, p, 0], learned_traj[-1, p, 1],
                    color=colors[p], marker='X', s=marker_size,
                    edgecolor='black', zorder=3)
        ax2.set_title('Learned Trajectories (Model Output)', fontsize=16)
        ax2.set_ylabel("Y Position", fontsize=14)
        ax2.set_xlabel("X Position", fontsize=14)
        # ax2.set_xlim(x_limits)  # Set y-axis limits
        # ax2.set_ylim(y_limits)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Noise-Free Trajectories (Ground Truth)
        for p in range(par):
            ax3.plot(clean_traj[:, p, 0], clean_traj[:, p, 1],
                    color=colors[p], linestyle=':', linewidth=line_width, alpha=alpha)
            ax3.scatter(clean_traj[0, p, 0], clean_traj[0, p, 1],
                    color=colors[p], marker='o', s=marker_size,
                    edgecolor='black', zorder=3)
            ax3.scatter(clean_traj[-1, p, 0], clean_traj[-1, p, 1],
                    color=colors[p], marker='X', s=marker_size,
                    edgecolor='black', zorder=3)
        ax3.set_title('Noise-Free Trajectories ', fontsize=16)
        ax3.set_xlabel("X Position", fontsize=14)
        ax3.set_ylabel("Y Position", fontsize=14)

        # ax3.set_xlim(x_limits)  # Set y-axis limits
        # ax3.set_ylim(y_limits)  # Set y-axis limits

        ax3.grid(True, alpha=0.3)
        
        # # Create unified legend
        legend_elements = []
        # for p in range(par):
        #     legend_elements.append(Line2D([0], [0], 
        #                             color=colors[p], 
        #                             lw=2, 
        #                             label=f'Particle {p+1}'))
        
        # # Add start/end markers to legend
        legend_elements.extend([
            Line2D([0], [0], marker='o', color='w', label='Start',
                markerfacecolor='black', markersize=10),
            Line2D([0], [0], marker='X', color='w', label='End',
                markerfacecolor='black', markersize=10)
        ])
        
        # Add line style indicators
        legend_elements.extend([
            Line2D([0], [0], color='k', linestyle='-', lw=2, label='Noisy'),
            Line2D([0], [0], color='k', linestyle='--', lw=2, label='Learned'),
            Line2D([0], [0], color='k', linestyle=':', lw=2, label='Noise-Free')
        ])
        
        fig.legend(handles=legend_elements, 
                loc='center right', 
                bbox_to_anchor=(1.15, 0.5),
                fontsize=14)
        
        # Adjust layout and save
        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.experiment}_trajectory_comparison_N{n_hidden}_X{n_x_neurons}_T{n_t_neurons}_{timestamp}.png"
        plt.savefig(f'results/{filename}', dpi=300, bbox_inches='tight')
        plt.close()

    def compare_noisy_and_clean(self, noisy_traj, clean_traj):
        os.makedirs('results', exist_ok=True)
        tp, par, dim = noisy_traj.shape
        
        # Create a figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        colors = plt.cm.tab10(np.linspace(0, 1, par))
        
        # Plot noisy trajectories on the left subplot
        for p in range(par):
            ax1.plot(noisy_traj[:, p, 0], noisy_traj[:, p, 1], 
                    color=colors[p], linestyle='-', 
                    linewidth=1.5, alpha=0.7,
                    label=f'Particle {p+1}')
            
            ax1.scatter(noisy_traj[0, p, 0], noisy_traj[0, p, 1], 
                    color=colors[p], marker='o', s=100, 
                    edgecolor='black', zorder=3)
            ax1.scatter(noisy_traj[-1, p, 0], noisy_traj[-1, p, 1], 
                    color=colors[p], marker='X', s=100, 
                    edgecolor='black', zorder=3)
        
        ax1.set_title('Noisy Trajectories', fontsize=14, pad=20)
        ax1.set_xlabel("X Position", fontsize=12)
        ax1.set_ylabel("Y Position", fontsize=12)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal', adjustable='datalim')
        
        # Plot clean trajectories on the right subplot
        for p in range(par):
            ax2.plot(clean_traj[:, p, 0], clean_traj[:, p, 1], 
                    color=colors[p], linestyle='-', 
                    linewidth=1.5, alpha=0.7,
                    label=f'Particle {p+1}')
            
            ax2.scatter(clean_traj[0, p, 0], clean_traj[0, p, 1], 
                    color=colors[p], marker='o', s=100, 
                    edgecolor='black', zorder=3)
            ax2.scatter(clean_traj[-1, p, 0], clean_traj[-1, p, 1], 
                    color=colors[p], marker='X', s=100, 
                    edgecolor='black', zorder=3)
        
        ax2.set_title('Clean Trajectories', fontsize=14, pad=20)
        ax2.set_xlabel("X Position", fontsize=12)
        ax2.set_ylabel("Y Position", fontsize=12)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal', adjustable='datalim')
        
        # Add a main title for the entire figure
        fig.suptitle(f'Trajectory Comparison: {self.experiment} Experiment', fontsize=16, y=1.02)
        
        plt.tight_layout()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results/{self.experiment}_comparison_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_means_and_covs(self, covs_datapoint_1, cov_dp_1, timepoints):
        os.makedirs('results', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        covs_data_point_1 = [item.cpu().detach().numpy() if isinstance(item, torch.Tensor) else item 
                        for item in covs_datapoint_1]
        
        covs_comparison_1 = [item.cpu().detach().numpy() if isinstance(item, torch.Tensor) else item 
                        for item in cov_dp_1]
      
        timepoints = [item.cpu().detach().numpy() if isinstance(item, torch.Tensor) else item 
                        for item in timepoints]
        

        plt.figure(figsize=(12, 5))  
        # # Set consistent y-axis limits for all plots
        if self.experiment == "Harmonic": 
            # y_limits = (0, 20)
            plt.plot(timepoints, covs_data_point_1, 'o-', markersize=6,linewidth=4, label='Model Covariance (Trace)', alpha=0.7)
            plt.plot(timepoints, covs_comparison_1, 'x-', markersize=6,linewidth=4, label='Analytical Covariance (Trace)', alpha=0.7)
            plt.xlabel('Time', fontsize=16)

            plt.ylabel('Covariance (Trace)', fontsize=16)
            plt.title('Covariance Comparison: Harmonic Experiment', fontsize=18)
            plt.legend(fontsize=14)
            plt.grid(True, alpha=0.3)
            # plt.ylim(y_limits)
            plt.tight_layout()
            
        else: 
            plt.plot(timepoints, covs_data_point_1, 'o-', markersize=4, label='Model Covariance (Trace)', alpha=0.7)
            plt.plot(timepoints, covs_comparison_1, 'x-', markersize=4, label='SDE Covariance (Trace)', alpha=0.7)

            plt.xlabel('Time', fontsize=16)
            plt.ylabel('Covariance', fontsize=16)
            plt.title('Covariance Comparison: Anharmonic Experiment', fontsize=18)
            # plt.ylim(y_limits)
            plt.legend(fontsize=14)
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            

        #change to pdf soon
        filename = f"results/{self.experiment}_covs_ts{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
            
    def train_flow_matching(self, clean_trajs, noisy_trajs):
        "learns the deterministic vector field"
        trainer = MeanFlowMatchingTrainer(model=self.network, opt = self.opt, clean_trajs=clean_trajs, noisy_trajs=noisy_trajs, batch_size=self.batch_size, n_epochs=self.n_epochs, experiment=self.experiment, step = self.step)
        self.training_entropy = trainer.train()
        return self.network

    def mean_flow_trajectory_simulator(self, initial_position, n_time_steps, ts, model):
        
        
        t_space = torch.linspace(0, 1, n_time_steps).to(Device)
        dt = 1 / (n_time_steps - 1)
        # Initialize outputs (all on same device)
        covs_datapoint_1 = torch.zeros(n_time_steps, device=Device)
        mean_divergence = torch.zeros(n_time_steps, device=Device)

        cov_comparison_1 = torch.zeros(n_time_steps, device=Device)
        entropy = torch.zeros(n_time_steps, device=Device)
        n_particles, dim = initial_position.shape

        positions = torch.zeros(n_time_steps, n_particles, dim, device=Device)
        positions[0] = initial_position
        


        single_particle_vel = lambda pos: model(
                x=pos.unsqueeze(0),
                t=t_space[0].unsqueeze(0), 
                r=(t_space[0] - dt).unsqueeze(0)
            ).squeeze(0)

            # Compute divergence for all particles
        div  = torch.func.vmap(
            lambda pos: torch.trace(torch.func.jacfwd(single_particle_vel)(pos))
        )(positions[0])

        mean_divergence[0] = torch.mean(div)
        
        if self.experiment == "Harmonic":
            cov_par = torch.trace(torch.cov(positions[0]))
            cov_dim = torch.trace(torch.cov(positions[0].T))
            covs_datapoint_1[0] = cov_par * cov_dim
            cov_comparison_1[0], entropy[0]= evaluate_analytical_cov(t_space[0])

        else:
            cov_par = torch.trace(torch.cov(positions[0]))
            cov_dim = torch.trace(torch.cov(positions[0].T))
            covs_datapoint_1[0] = cov_par * cov_dim
            # print(f'model cov: {cov_par * cov_dim}')
            cov_comparison_1[0] =  evaluate_sde_cov(0, n_time_steps, self.noisy_trajs).to(Device)

        for n in range(n_time_steps - 1):
            current_t = t_space[n+1]
            positions_n = positions[n].clone().requires_grad_(True)
            
            # Compute velocity
            vel = model(
                x=positions_n,
                t=t_space[n].expand(n_particles, 1),
                r=(t_space[n] - dt).expand(n_particles, 1)
            )

            single_particle_vel = lambda pos: model(
                x=pos.unsqueeze(0),
                t=t_space[n+1].unsqueeze(0), 
                r=(t_space[n+1] - dt).unsqueeze(0)
            ).squeeze(0)

            # Compute divergence for all particles
            div  = torch.func.vmap(
                lambda pos: torch.trace(torch.func.jacfwd(single_particle_vel)(pos))
            )(positions_n)

            mean_divergence[n+1] = torch.mean(div)

            positions[n+1] = (positions_n + dt * vel).detach()

            if self.experiment == "Harmonic":
                current_pos = positions[n+1]
                cov_par = torch.trace(torch.cov(current_pos))
                cov_dim = torch.trace(torch.cov(current_pos.T))
                covs_datapoint_1[n+1] = cov_par * cov_dim
                cov_comparison_1[n+1], entropy[n+1] = evaluate_analytical_cov(current_t)
            else:
                current_pos = positions[n+1]
                cov_par = torch.trace(torch.cov(current_pos))
                # cov_dim = torch.trace(torch.cov(current_pos.T))
                covs_datapoint_1[n+1] = cov_par
                # print(f'model cov: {cov_par * cov_dim}')
                # covs_datapoint_1[n+1] = torch.trace(cov_matrix)
                sde_cov = evaluate_sde_cov(n, n_time_steps, self.noisy_trajs).to(Device)

                cov_comparison_1[n+1] =  sde_cov


                # sde_cov = evaluate_sde_cov(n, n_time_steps, self.noisy_trajs)
                # cov_comparison_1[n+1] = torch.trace(sde_cov)

        return positions, covs_datapoint_1,cov_comparison_1, t_space, entropy, mean_divergence