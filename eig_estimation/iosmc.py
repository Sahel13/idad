import torch
from torch.distributions import MultivariateNormal, Categorical
from eig_estimation.ibis import ibis_step


class MultivariateLogNormal:
    def __init__(self, mean_vector, covariance_matrix):
        self.mean = mean_vector
        self.cov = covariance_matrix
        self.log_dist = MultivariateNormal(self.mean, self.cov)

    def sample(self, shape):
        return torch.exp(self.log_dist.sample(shape))

    # TODO: Verify if this is correct.
    def log_prob(self, x):
        return self.log_dist.log_prob(torch.log(x)) / torch.prod(
            x, dim=-1, keepdim=True
        )


class StateStruct:
    def __init__(self, init_state, nb_steps, nb_trajectories):
        self.state_dim = init_state.shape[0]
        self.trajectories = torch.zeros(nb_trajectories, nb_steps + 1, self.state_dim)
        self.trajectories[:, 0, :] = init_state
        self.cumulative_return = torch.zeros(nb_trajectories)


class ParamStruct:
    def __init__(self, param_prior, nb_particles, nb_trajectories):
        self.param_prior = param_prior
        self.nb_particles = nb_particles
        param_dim = len(param_prior.loc)
        self.particles = torch.zeros(nb_trajectories, nb_particles, param_dim)
        self.weights = torch.ones(nb_trajectories, nb_particles) / nb_particles
        self.log_weights = torch.zeros(nb_trajectories, nb_particles)
        self.log_likelihoods = torch.zeros(nb_trajectories, nb_particles)

        for n in range(nb_trajectories):
            self.particles[n, :, :] = param_prior.sample((nb_particles,))
            self.log_likelihoods[n, :] = self.param_prior.log_prob(
                self.particles[n, :, :]
            )


class IBISDynamics:
    def __init__(self, xdim, udim, step, diffusion_vector):
        self.xdim = xdim
        self.udim = udim
        self.step = step
        self.diffusion_vector = diffusion_vector
        self.noise_dist = MultivariateNormal(
            torch.zeros(self.xdim),
            scale_tril=torch.diag(
                torch.sqrt(torch.square(self.diffusion_vector) * self.step + 1e-8)
            ),
        )

    def drift_fn(self, p, x, u):
        raise NotImplementedError

    def diffusion_fn(self):
        raise NotImplementedError

    def conditional_mean(self, p, x, u):
        return x + self.step * self.drift_fn(p, x, u)

    def conditional_sample(self, ps, xs, us):
        new_xs = torch.zeros_like(xs)
        for i in range(xs.shape[0]):
            new_xs[i] = self.conditional_mean(ps[i], xs[i], us[i])
        return new_xs + self.noise_dist.sample((xs.shape[0],))

    def conditional_logpdf(self, ps, x, u, xn):
        batch_size = ps.shape[0]
        mean_vals = torch.zeros((batch_size, self.xdim))
        for i in range(batch_size):
            mean_vals[i] = self.conditional_mean(ps[i], x, u)
        return self.noise_dist.log_prob(xn - mean_vals)

    def info_gain_increment(self, ps, lws, x, u, xn):
        lls = self.conditional_logpdf(ps, x, u, xn)
        mod_lws = torch.nn.Softmax(dim=0)(lws + lls)
        return (
            torch.dot(mod_lws, lls)
            - torch.logsumexp(lls + lws, dim=0)
            + torch.logsumexp(lws, dim=0)
        )


class ClosedLoop:
    # def __init__(self, dynamics: IBISDynamics, policy: LSTMImplicitDAD):
    def __init__(self, dynamics: IBISDynamics, policy, scale: float, shift: float):
        self.dynamics = dynamics
        self.policy = policy
        self.scale = scale
        self.shift = shift

    def transform_designs(self, xi_untransformed):
        return self.shift + self.scale * torch.nn.Tanh()(xi_untransformed)

    def sample_conditional(self, ps, trajectories):
        # Sample actions first.
        list = [(trajectories[:, t, self.dynamics.xdim :], trajectories[:, t, : self.dynamics.xdim]) for t in range(trajectories.shape[1])]
        untransformed_us = self.policy(*list)
        us = self.transform_designs(untransformed_us)
        # us = self.policy(trajectories)
        # Sample states.
        xs = trajectories[:, -1, 0 : self.dynamics.xdim]
        xns = self.dynamics.conditional_sample(ps, xs, us)
        # iDAD policies take the untransformed designs as input.
        return torch.cat((xns, untransformed_us), dim=-1)
        # return torch.cat((xns, us), dim=-1)

    def sample_marginal(self, ps, ws, trajectories):
        nb_trajectories, _, param_dim = ps.shape
        resampled_ps = torch.zeros(nb_trajectories, param_dim)
        for n in range(nb_trajectories):
            idx = Categorical(ws[n]).sample()
            resampled_ps[n] = ps[n, idx, :]

        return self.sample_conditional(resampled_ps, trajectories)


def estimate_eig(
    nb_steps: int,
    nb_trajectories: int,
    nb_particles: int,
    param_prior,
    init_state,
    closed_loop: ClosedLoop,
    nb_moves: int = 3,
):
    # Initialize structs.
    state_struct = StateStruct(init_state, nb_steps, nb_trajectories)
    param_struct = ParamStruct(param_prior, nb_particles, nb_trajectories)
    for t in range(nb_steps):
        # Propagate particles.
        state_struct.trajectories[:, t + 1, :] = closed_loop.sample_marginal(
            param_struct.particles,
            param_struct.weights,
            state_struct.trajectories[:, 0 : t + 1, :],
        )
        for n in range(nb_trajectories):
            # Compute the info gain increment and update.
            xdim = closed_loop.dynamics.xdim
            x = state_struct.trajectories[n, t, 0:xdim]
            u = state_struct.trajectories[n, t + 1, xdim:]
            xn = state_struct.trajectories[n, t + 1, 0:xdim]
            state_struct.cumulative_return[
                n
            ] += closed_loop.dynamics.info_gain_increment(
                param_struct.particles[n], param_struct.log_weights[n], x, u, xn
            )

            # IBIS step.
            ibis_step(
                t,
                n,
                state_struct.trajectories[n, 0 : t + 2, :],
                closed_loop.dynamics,
                param_prior,
                nb_moves,
                param_struct,
            )
        print("Step: ", t, "Return: ", torch.mean(state_struct.cumulative_return))
    return torch.mean(state_struct.cumulative_return)
