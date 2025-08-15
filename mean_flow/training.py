import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from abc import ABC, abstractmethod
from typing import Optional
from torch.nn.utils import clip_grad_norm_



class Trainer(ABC):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    @abstractmethod
    def get_train_loss(self, **kwargs) -> torch.Tensor:
        pass


    def train(self, num_epochs: int, device: torch.device, **kwargs) -> torch.Tensor:
        # Start
        self.model.to(device)
        opt = self.opt
        self.model.train()


        pbar = tqdm(enumerate(range(num_epochs)))
        for idx, epoch in pbar:
            opt.zero_grad()
            loss = self.get_train_loss(epoch, **kwargs)
            loss.backward()
            opt.step()
            pbar.set_description(f'Epoch {idx}, loss: {loss.item()}')

        self.model.eval()




        
class MeanFlowMatchingTrainer(Trainer):
    def __init__(self, 
                 opt,
                 model: torch.nn.Module, 
                 clean_trajs: list,
                 noisy_trajs: list,
                 batch_size: int,
                 n_epochs: int,
                 experiment: str,
                 step: int
                ):
        
        super().__init__(model)
        self.clean_trajs = clean_trajs
        self.noisy_trajs = noisy_trajs
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.opt = opt
        self.n_epochs = n_epochs
        self.experiment = experiment
        self.step = step
    def sample_t_and_r_indices(self, batch_size, step, epoch):

        progress = epoch / self.n_epochs
        high = int(2500 * (1- progress) + 5000 * progress)

        if epoch % 50 == 0: 
            s = -1 
            t = torch.randint(low=0, high=high-step, size=(batch_size,))
            r = torch.clamp(t + step, max=5000)  #prevent indices above maximum
        else: 
            s = 1
            t = torch.randint(low=step, high=high, size=(batch_size,))
            r = torch.clamp(t - step, min=0)  # Prevent negative indices

        return t.unsqueeze(1), r.unsqueeze(1), s


    def get_space_and_time_for_idx(self, t_indices, r_indices, trajs): 
        train_trajs_tensor = torch.tensor(trajs)  
        
        # Extract positions and times
        x_t = train_trajs_tensor[t_indices.long(), :, :2].squeeze(1).float().to(self.device)  
        x_r = train_trajs_tensor[r_indices.long(), :, :2].squeeze(1).float().to(self.device)  
        tau_t = train_trajs_tensor[t_indices.long(), :, 2:3].squeeze(1).float().to(self.device)  
        tau_r = train_trajs_tensor[r_indices.long(), :, 2:3].squeeze(1).float().to(self.device)  
       
        return x_t, tau_t, x_r, tau_r

    def get_train_loss(self, epoch: int, **kwargs) -> torch.Tensor:

        if self.experiment == "Harmonic":
            if epoch < self.n_epochs // 3: 
                trajs = self.clean_trajs
            elif  self.n_epochs // 3 < epoch <  self.n_epochs // 2:
                if epoch % 2 == 0: 
                    trajs = self.clean_trajs
                else: 
                    trajs = self.noisy_trajs
            else: 
                trajs = self.noisy_trajs


        if self.experiment == "Anharmonic":
            if epoch <= self.n_epochs // 5: 
                trajs = self.clean_trajs
            elif  self.n_epochs // 5 < epoch <   self.n_epochs // 2:
                if epoch % 2 == 0: 
                    trajs = self.clean_trajs
                else: 
                    trajs = self.noisy_trajs
            else:
                trajs = self.noisy_trajs

        if self.experiment == "Keller_Segel":
            if epoch <= self.n_epochs // 2: 
                trajs = self.clean_trajs
            elif  self.n_epochs // 2 < epoch <   3 * self.n_epochs // 4:
                if epoch % 2 == 0: 
                    trajs = self.clean_trajs
                else: 
                    trajs = self.noisy_trajs
            else:
                trajs = self.noisy_trajs

        

        t_idx, r_idx, s = self.sample_t_and_r_indices(batch_size=self.batch_size, epoch=epoch, step = self.step)
        x_t, tau_t, x_r, tau_r = self.get_space_and_time_for_idx(t_idx, r_idx, trajs)
        
        ep = torch.rand_like(tau_t)
        t_var = tau_r + ep * (tau_t - tau_r)
        alpha = (t_var - tau_r) / (tau_t - tau_r + 1e-8)
        interpolated_samples = x_r * (1 - alpha) + x_t * alpha
        velocity = (x_t - x_r) / (tau_t - tau_r + 1e-8)
        
        def full_forward(x, t, r):
            return self.model.forward(x, t, r)
        
        primal_inputs = (interpolated_samples, t_var, tau_r)
        vectors = (torch.zeros_like(interpolated_samples), 
                torch.ones_like(t_var), 
                torch.zeros_like(tau_r))
        
        u, dudt = torch.func.jvp(full_forward, primal_inputs, vectors)
        u_tgt = velocity + s * torch.norm(tau_t - tau_r) * dudt.detach()
        

        mse_loss = torch.nn.functional.mse_loss(u, u_tgt)

        total_loss =  mse_loss


        return total_loss



    def train(self, **kwargs):


        initial_clip = 1.0
        final_clip = 2.0
        clip_ramp_epochs = (self.n_epochs // 2) 
        self.model.to(self.device)
        opt = self.opt
        
        with tqdm(range(self.n_epochs), desc="Training") as pbar:
            for epoch in pbar:
                # Dynamic gradient clipping
                if epoch < clip_ramp_epochs:
                    current_clip = initial_clip + (final_clip - initial_clip) * (epoch/clip_ramp_epochs)
                else:
                    current_clip = final_clip
                
                opt.zero_grad()
                loss = self.get_train_loss(epoch, **kwargs)
                loss.backward()
            
                total_norm = clip_grad_norm_(
                    self.model.parameters(), 
                    max_norm=current_clip,
                    norm_type=2.0
                )
                pbar.set_postfix(loss=loss.item(), 
                            grad_norm=total_norm.item(),
                            clip=current_clip)
                opt.step()

        self.model.eval()
