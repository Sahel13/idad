import torch
from torch.distributions import MultivariateNormal, Uniform
from iosmc import IBISDynamics, ClosedLoop, estimate_eig


class Pendulum(IBISDynamics):
    def __init__(self):
        xdim = 2
        udim = 1
        step = 0.05
        diffusion_vector = torch.tensor([0.1, 0.1])
        super().__init__(xdim, udim, step, diffusion_vector)

    def drift_fn(self, p, x, u):
        p1, p2, p3 = p
        q, dq = x
        ddq = -torch.sin(q) * p1 - dq * p2 + u * p3
        return torch.tensor([dq, ddq])


def random_policy(trajectories):
    nb_trajectories = trajectories.shape[0]
    u_dim = 1
    return Uniform(-1.0, 1.0).sample((nb_trajectories,1))


scale, shift = 1.0, 0.0
closed_loop = ClosedLoop(Pendulum(), random_policy, scale, shift)

param_prior = MultivariateNormal(
    torch.tensor([14.7, 0.0, 3.0]), torch.diag(torch.tensor([0.1, 0.01, 0.1]))
)
init_state = torch.zeros(3)
nb_steps = 50
nb_trajectories = 16
nb_particles = 1024

estimate = estimate_eig(nb_steps, nb_trajectories, nb_particles, param_prior, init_state, closed_loop)
