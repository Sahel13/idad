import torch
from test_linear_pendulum_random import LinearPendulum
from torch.distributions import MultivariateNormal
from iosmc import StateStruct, ParamStruct
from ibis import ibis_step, cov


def simulate_trajectories(
    nb_trajectories,
    nb_steps,
    init_state,
    true_params,
    param_prior,
):
    dynamics = LinearPendulum()
    xdim = dynamics.xdim
    udim = dynamics.udim
    h = dynamics.step
    trajectories = torch.zeros(nb_trajectories, nb_steps + 1, xdim + udim)

    posterior_means = param_prior.loc.repeat(nb_trajectories, 1)
    posterior_covs = param_prior.covariance_matrix.repeat(nb_trajectories, 1, 1)

    trajectories[:, 0, :] = init_state
    for t in range(nb_steps):
        zs = trajectories[:, t]
        us = torch.ones(nb_trajectories, udim)  # Just to test.
        xns = dynamics.conditional_sample(true_params, zs[:, :xdim], us)
        trajectories[:, t + 1, :] = torch.cat((xns, us), dim=-1)

        # Update the posteriors.
        for n in range(nb_trajectories):
            z = zs[n]
            A = torch.tensor([[1.0, h], [0.0, 1.0]])
            B = torch.tensor([0.0, h]).reshape(2, 1) @ torch.tensor(
                [-torch.sin(z[0]), -z[1], z[2]]
            ).reshape(1, 3)
            x_hat = A @ z[:xdim] + B @ posterior_means[n]
            x_diff = xns[n] - x_hat
            S = B @ posterior_covs[n] @ B.T + dynamics.noise_dist.covariance_matrix
            G = posterior_covs[n] @ torch.linalg.solve(S, B).T
            posterior_means[n] += G @ x_diff
            posterior_covs[n] -= G @ S @ G.T

    return trajectories, posterior_means, posterior_covs


init_state = torch.tensor([0.0, 0.0, 0.0])
param_prior = MultivariateNormal(
    torch.tensor([14.7, 0.0, 3.0]), torch.diag(torch.tensor([0.1, 0.01, 0.1]))
)
true_params = torch.tensor([13.7, 0.01, 4.0]).repeat(2, 1)
trajectories, posterior_means, posterior_covs = simulate_trajectories(
    2, 50, init_state, true_params, param_prior
)


def ibis(trajectories, nb_particles, param_prior, init_state, dynamics, nb_moves):
    nb_trajectories = trajectories.shape[0]
    nb_steps = trajectories.shape[1] - 1
    # Initialize structs.
    state_struct = StateStruct(init_state, nb_steps, nb_trajectories)
    param_struct = ParamStruct(param_prior, nb_particles, nb_trajectories)
    # Copy over the trajectory to the state struct.
    state_struct.trajectories = trajectories

    for t in range(nb_steps):
        for n in range(nb_trajectories):
            # IBIS step.
            ibis_step(
                t,
                n,
                state_struct.trajectories[n, 0 : t + 2, :],
                dynamics,
                param_prior,
                nb_moves,
                param_struct,
            )
        print("Step: ", t)
    return param_struct


param_struct = ibis(trajectories, 128, param_prior, init_state, LinearPendulum(), 1)


def weighted_mean(tensor, weights):
    w_sum = weights.sum(dim=-1, keepdim=True)
    mean = (weights * tensor).sum(dim=-1, keepdim=True) / w_sum
    return mean


for n in range(2):
    print("Trajectory: ", n)
    print("True params: ", true_params[n])
    print("Closed form mean: ", posterior_means[n])
    print(
        "IBIS mean: ",
        weighted_mean(param_struct.particles[n, :, :].T, param_struct.weights[n]),
    )
    print("Closed form cov: ", posterior_covs[n])
    print(
        "IBIS cov: ",
        cov(
            param_struct.particles[n, :, :],
            rowvar=False,
            weights=param_struct.weights[n],
        ),
    )
