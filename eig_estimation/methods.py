import torch
from torch.distributions import MultivariateNormal, Categorical
from neural.aggregators import LSTMImplicitDAD


class StateStruct:
    def __init__(self, init_state, nb_steps, nb_trajectories):
        self.state_dim = init_state.shape[0]
        self.trajectories = torch.zeros(nb_trajectories, nb_steps + 1, self.state_dim)
        self.trajectories[:, 0, :] = init_state
        self.cumulative_return = torch.zeros(nb_trajectories)


class ParamStruct:
    def __init__(self, param_prior, nb_steps, nb_particles, nb_trajectories):
        self.param_dim = len(param_prior.loc)
        self.particles = torch.zeros(nb_trajectories, nb_particles, self.param_dim)
        # TODO: Check if the sampling is correct.
        for i in range(nb_trajectories):
            self.particles[i, :, :] = param_prior.sample((nb_particles,))
        self.weights = torch.ones(nb_trajectories, nb_particles) / nb_particles
        self.log_weights = torch.zeros(nb_trajectories, nb_particles)
        self.log_likelihoods = torch.zeros(nb_trajectories, nb_particles)


class IBISDynamics:
    def __init__(self, xdim, udim, step, diffusion_vector):
        self.xdim = xdim
        self.udim = udim
        self.step = step
        self.diffusion_vector = diffusion_vector
        self.conditional_dynamics_covar = (
            torch.square(self.diffusion_vector) * self.step + 1e-8
        )

    def drift_fn(self, p, x, u):
        raise NotImplementedError

    def diffusion_fn(self):
        raise NotImplementedError

    def conditional_dynamics_mean(self, p, x, u):
        return x + self.step * self.drift_fn(p, x, u)

    def conditional_dynamics_sample(self, ps, xs, us):
        new_xs = torch.zeros_like(xs)
        for i in range(xs.shape[0]):
            new_xs[i] = self.conditional_dynamics_mean(
                ps[i], xs[i], us[i]
            ) + torch.normal(torch.zeros(self.xdim), self.conditional_dynamics_covar)
        return new_xs

    def conditional_dynamics_logpdf(self, ps, x, u, xn):
        batch_size = ps.shape[0]
        logpdfs = torch.zeros(batch_size)
        for i in range(batch_size):
            mean = self.conditional_dynamics_mean(ps[i], x, u)
            logpdfs[i] = MultivariateNormal(
                mean,
                scale_tril=torch.diag(torch.sqrt(self.conditional_dynamics_covar)),
            ).log_prob(xn)
        return logpdfs

    def marginal_dynamics_logpdf(self, ps, lws, x, u, xn):
        batch_size = ps.shape[0]
        cond_logpdfs = torch.zeros(batch_size)
        for i in range(batch_size):
            cond_logpdfs[i] = self.conditional_dynamics_logpdf(ps[i], x, u, xn)

        return torch.logsumexp(cond_logpdfs + lws) - torch.logsumexp(lws)

    def info_gain_increment(self, ps, lws, x, u, xn):
        lls = self.conditional_dynamics_logpdf(ps, x, u, xn)
        mod_lws = torch.nn.Softmax(dim=0)(lws + lls)
        return (
            torch.dot(mod_lws, lls) - torch.logsumexp(lls + lws, dim=0) + torch.logsumexp(lws, dim=0)
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

    def conditional_dynamics_sample(self, ps, trajectories):
        # Sample actions first.
        # TODO: This needs to be checked, the action might need to be the first input variable.
        # us = self.policy.lazy(trajectories)
        untransformed_us = self.policy(trajectories)
        us = self.transform_designs(untransformed_us)
        # Sample states.
        xs = trajectories[:, -1, 0 : self.dynamics.xdim]
        xns = self.dynamics.conditional_dynamics_sample(ps, xs, us)
        # iDAD policies take the untransformed designs as input.
        return torch.cat((xns, untransformed_us), dim=-1)

    def marginal_dynamics_sample(self, ps, ws, trajectories):
        nb_trajectories, _, param_dim = ps.shape
        resampled_ps = torch.zeros(nb_trajectories, param_dim)
        for n in range(nb_trajectories):
            idx = Categorical(ws[n]).sample()
            resampled_ps[n] = ps[n, idx, :]

        return self.conditional_dynamics_sample(resampled_ps, trajectories)


def iosmc2(
    nb_steps: int,
    nb_trajectories: int,
    nb_particles: int,
    param_prior,
    init_state,
    closed_loop: ClosedLoop,
):
    # Initialize structs.
    state_struct = StateStruct(init_state, nb_steps, nb_trajectories)
    param_struct = ParamStruct(param_prior, nb_steps, nb_particles, nb_trajectories)
    for t in range(nb_steps):
        # Propagate particles.
        trajectories = closed_loop.marginal_dynamics_sample(
            param_struct.particles,
            param_struct.weights,
            state_struct.trajectories[:, 0:t+1, :],
        )
        # state_struct.trajectories[:, t + 1, :] = closed_loop.marginal_dynamics_sample(
        #     param_struct.particles,
        #     param_struct.weights,
        #     state_struct.trajectories[:, 0:t+1, :],
        # )
        # Compute the info gain increment and update.
        for n in range(nb_trajectories):
            xdim = closed_loop.dynamics.xdim
            x = state_struct.trajectories[n, t, 0:xdim]
            u = state_struct.trajectories[n, t, xdim:]
            xn = state_struct.trajectories[n, t + 1, 0:xdim]
            state_struct.cumulative_return[
                n
            ] += closed_loop.dynamics.info_gain_increment(
                param_struct.particles[n], param_struct.log_weights[n], x, u, xn
            )
        # TODO: IBIS step.
    return None


# def ibis_step():
#     """
#     Single step of IBIS.
#     """
#     # Reweight
#     # if ess < threshold:
#     # Resample
#     # Move
#     pass
