import torch
from torch.distributions import MultivariateNormal


def inverse_cdf(uniforms, weights):
    """Inverse CDF sampling."""
    idx = torch.zeros_like(uniforms, dtype=torch.int64)
    i = 0
    cumsum = weights[0]
    for m in range(len(uniforms)):
        while uniforms[m] > cumsum:
            if i < len(weights) - 1:
                i += 1
                cumsum += weights[i]
            else:
                break
        idx[m] = i
    return idx


def systematic_resampling(weights):
    """Systematic resampling."""
    nb_particles = len(weights)
    ordered_uniforms = (torch.arange(nb_particles) + torch.rand(1)) / nb_particles
    return inverse_cdf(ordered_uniforms, weights)


def ess(weights):
    """Effective sample size."""
    return 1.0 / torch.sum(torch.square(weights))


def cov(tensor, weights):
    """
    Empirical covariance of weighted samples.
    """
    fact = 1 / weights.sum()
    emp_mean = fact * torch.einsum("i, id->d", weights, tensor)
    diff = tensor - emp_mean
    emp_cov = torch.einsum("i, id, ik -> dk", weights, diff, diff)
    return fact * emp_cov


def gaussian_proposal_fn(particles, weights):
    covar = cov(particles, weights=weights)
    eig_vals, eig_vecs = torch.symeig(covar, eigenvectors=True)
    sqrt_eig_vals = torch.sqrt(torch.maximum(eig_vals, torch.tensor(1e-8)))
    sqrt_covar = eig_vecs @ torch.diag(sqrt_eig_vals)
    return particles + torch.randn_like(particles) @ sqrt_covar.T


def lognormal_proposal_fn(particles, weights):
    log_particles = torch.log(particles)
    covar = cov(log_particles, weights=weights)
    eig_vals, eig_vecs = torch.symeig(covar, eigenvectors=True)
    sqrt_eig_vals = torch.sqrt(torch.maximum(eig_vals, torch.tensor(1e-8)))
    sqrt_covar = eig_vecs @ torch.diag(sqrt_eig_vals)
    log_particles += torch.randn_like(log_particles) @ sqrt_covar.T
    return torch.exp(log_particles)


def accumulate_likelihood(trajectory, particles, dynamics, param_prior):
    lls = param_prior.log_prob(particles)
    for t in range(trajectory.shape[0] - 1):
        x = trajectory[t, : dynamics.xdim]
        u = trajectory[t + 1, dynamics.xdim :]
        xn = trajectory[t + 1, : dynamics.xdim]
        lls += dynamics.conditional_logpdf(particles, x, u, xn)
    return lls


def kernel(n, trajectory, dynamics, param_struct):
    if isinstance(param_struct.param_prior, MultivariateNormal):
        prop_particles = gaussian_proposal_fn(
            param_struct.particles[n], param_struct.weights[n]
        )
    else:
        prop_particles = lognormal_proposal_fn(
            param_struct.particles[n], param_struct.weights[n]
        )

    prop_log_likelihood = accumulate_likelihood(
        trajectory, prop_particles, dynamics, param_struct.param_prior
    )
    log_rvs = torch.log(torch.rand(param_struct.nb_particles))

    for m in range(param_struct.nb_particles):
        if log_rvs[m] < prop_log_likelihood[m] - param_struct.log_likelihoods[n, m]:
            param_struct.particles[n, m] = prop_particles[m]
            param_struct.log_likelihoods[n, m] = prop_log_likelihood[m]

    return None


def move(n, trajectory, dynamics, nb_moves, param_struct):
    for _ in range(nb_moves):
        kernel(n, trajectory, dynamics, param_struct)
    return None


def reweight(t, n, trajectory, dynamics, param_struct):
    # Get the log weight increments.
    log_weight_increments = dynamics.conditional_logpdf(
        param_struct.particles[n],
        trajectory[t, : dynamics.xdim],
        trajectory[t + 1, dynamics.xdim :],
        trajectory[t + 1, : dynamics.xdim],
    )
    param_struct.log_weights[n] += log_weight_increments
    param_struct.log_likelihoods[n] += log_weight_increments
    param_struct.weights[n] = torch.nn.Softmax(dim=0)(param_struct.log_weights[n])
    return None


def ibis_step(t, n, trajectory, dynamics, nb_moves, param_struct):
    # Reweight.
    reweight(t, n, trajectory, dynamics, param_struct)

    if ess(param_struct.weights[n]) < 0.75 * param_struct.nb_particles:
        # Resample.
        idx = systematic_resampling(param_struct.weights[n])
        param_struct.particles[n, :, :] = param_struct.particles[n, idx, :]
        param_struct.weights[n] = (
            torch.ones(param_struct.nb_particles) / param_struct.nb_particles
        )
        param_struct.log_weights[n] = torch.zeros(param_struct.nb_particles)
        param_struct.log_likelihoods[n] = param_struct.log_likelihoods[n, idx]
        # Move.
        move(n, trajectory, dynamics, nb_moves, param_struct)
    return None
