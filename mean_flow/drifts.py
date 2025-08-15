"""
Nicholas M. Boffi
7/29/22

Drift terms for score-based transport modeling.
"""

from typing import Callable, Tuple
import torch 
import numpy as np

compute_particle_diffs = torch.vmap(
        torch.vmap(lambda x, y: x - y, in_dims=(0, None), out_dims=0),
        in_dims=(None, 0), out_dims=1
    )



def harmonic_trap(
    x: np.ndarray,
    t: float,
    compute_mut: Callable[[float], np.ndarray],
    N: int,
    d: int
) -> np.ndarray:
    """Forcing for particles in a harmonic trap with harmonic repulsion."""
    mut = compute_mut(t)
    particle_pos = x.reshape((N, d))
    particle_forces = -0.5*(particle_pos + mut[None, :] \
            + np.mean(particle_pos, axis=0)[None, :])

    return particle_forces.ravel()



def gaussian_interaction(xs: np.ndarray, A: float, r: float, N: int):
    particle_diffs = xs[:, None, :] - xs[None, :, :]
    dist_sq = np.sum(particle_diffs**2, axis=2)
    gauss_facs = np.exp(-dist_sq/(1.5*r**2))  # Wider interaction range
    return (2*A/(N*r**2)) * np.sum(particle_diffs * gauss_facs[:, :, None], axis=1)


def anharmonic_gaussian(
    x: np.ndarray,
    t: float,
    A: float,
    r: float,
    B: float,
    N: int,
    d: int,
    compute_mut: Callable[[float], np.ndarray]
) -> np.ndarray:
    """Corrected implementation matching the SDE"""
    particle_pos = x.reshape((N, d))
    beta_t = compute_mut(t)
    diff = particle_pos - beta_t[None, :]
    diff_norms_sq = np.sum(diff**2, axis=1)
    
    # Trap force (note 4B factor)
    trap_force = -4 * B * diff * diff_norms_sq[:, None]
    
    # Interaction force
    interaction = gaussian_interaction(particle_pos, A, r, N)
    
    return (trap_force + interaction).ravel()

def anharmonic(
    x: np.ndarray,
    t: float,
    compute_mut: Callable[[float], np.ndarray]
) -> np.ndarray:
    """Single particle in an anharmonic trap."""
    diff = x - compute_mut(t)
    return -diff * (diff @ diff)


def anharmonic_harmonic(
    x: np.ndarray,
    t: float,
    A: float,
    B: float,
    N: int,
    d: int,
    compute_mut: Callable[[float], np.ndarray],
) -> np.ndarray:
    """Harmonically-interacting particles in an anharmonic trap."""
    particle_pos = x.reshape((N, d))
    diff = particle_pos - compute_mut(t)[None, :]
    diff_norms = np.sum(diff**2, axis=1)
    xbar = np.mean(particle_pos, axis=0)
    return np.ravel(-B*diff*diff_norms[:, None] + A*(particle_pos - xbar[None, :]))



def keller_segel(x, t, chi, r, N, d=2):
    particle_pos = x.reshape((N, d))
    diffs = particle_pos[:, None, :] - particle_pos[None, :, :]
    dist_sq = np.sum(diffs**2, axis=2)
    gaussian = np.exp(-dist_sq / (2 * r**2))
    
    # Flip the sign here - particles should move toward concentration peaks
    grad_c = -np.sum(diffs * gaussian[..., None] / r**2, axis=1)
    drift = chi * grad_c
    
    return drift.ravel()

# def keller_segel(
#     x: np.ndarray,
#     t: float,
#     chi: float,      # Chemotactic sensitivity
#     r: float,        # Interaction radius
#     N: int,          # Number of particles
#     d: int = 2       # Dimension (must be 2 for this implementation)
# ) -> np.ndarray:
#     """
#     Computes the Keller-Segel drift for particles in 2D.
#     """
#     assert d == 2, "This implementation is for 2D only."
#     particle_pos = x.reshape((N, d))
    
#     # Compute pairwise differences and distances
#     diffs = particle_pos[:, None, :] - particle_pos[None, :, :]  # Shape (N, N, 2)
#     dist_sq = np.sum(diffs**2, axis=2)
    
#     # Compute normalized Gaussian kernel and its gradient
#     gaussian = np.exp(-dist_sq / (2 * r**2)) / (2 * np.pi * r**2)  # Normalized
    
#     # Gradient of concentration field (sum over all other particles)
#     grad_c = np.sum(diffs * gaussian[..., None] / r**2, axis=1)  # Removed the minus sign
    
#     # Keller-Segel drift: chi * grad_c
#     drift = chi * grad_c
    
#     return drift.ravel()





# def keller_segel(
#     x: np.ndarray,
#     t: float,
#     chi: float,      # Chemotactic sensitivity
#     r: float,        # Interaction radius
#     N: int,          # Number of particles
#     d: int = 2       # Dimension (must be 2 for this implementation)
# ) -> np.ndarray:
#     """
#     Computes the Keller-Segel drift for particles in 2D.
#     Args:
#         x: Flattened array of particle positions, shape (N*d,).
#         t: Time (unused here, but kept for consistency with other drifts).
#         chi: Chemotactic sensitivity.
#         r: Interaction radius.
#         N: Number of particles.
#         d: Dimension (must be 2).
#     Returns:
#         Flattened array of drift forces, shape (N*d,).
#     """
#     assert d == 2, "This implementation is for 2D only."
#     particle_pos = x.reshape((N, d))  # Shape (N, 2)

#     # Compute pairwise differences and distances
#     diffs = particle_pos[:, None, :] - particle_pos[None, :, :]  # Shape (N, N, 2)
#     dist_sq = np.sum(diffs**2, axis=2)  # Shape (N, N)

#     # Compute Gaussian kernel and its gradient
#     gaussian = np.exp(-dist_sq / (2 * r**2))  # Shape (N, N)
#     grad_c = -np.sum(diffs * gaussian[..., None] / r**2, axis=1)  # Shape (N, 2)

#     # Keller-Segel drift: chi * grad_c
#     drift = chi * grad_c  # Shape (N, 2)

#     return drift.ravel()  # Flatten to (N*d,)