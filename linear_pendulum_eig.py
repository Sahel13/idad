import torch
from torch.distributions import MultivariateNormal, Uniform
from eig_estimation.iosmc import IBISDynamics, ClosedLoop, estimate_eig
import argparse
import mlflow
from experiment_tools.output_utils import get_mlflow_meta


class LinearPendulum(IBISDynamics):
    def __init__(self):
        xdim = 2
        udim = 1
        step = 0.05
        diffusion_vector = torch.tensor([0.0, 0.1])
        super().__init__(xdim, udim, step, diffusion_vector)

    def drift_fn(self, p, x, u):
        p1, p2, p3 = p
        q, dq = x
        ddq = -torch.sin(q) * p1 - dq * p2 + u * p3
        return torch.tensor([dq, ddq])


# def random_policy(trajectories):
#     nb_trajectories = trajectories.shape[0]
#     return Uniform(-1.0, 1.0).sample((nb_trajectories,1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", type=str)
    args = parser.parse_args()

    # Load the trained policy.
    filter_string = "params.status='complete'"
    meta = get_mlflow_meta(experiment_id=args.experiment_id, filter_string=filter_string)
    experiment_run_ids = [run.info.run_id for run in meta]
    run_id = experiment_run_ids[0]
    artifact_path = f"mlruns/{args.experiment_id}/{run_id}/artifacts"
    model_location = f"{artifact_path}/model"
    trained_model = mlflow.pytorch.load_model(model_location, map_location="cpu")
    idad_policy = trained_model.design_net

    scale, shift = 1.0, 0.0
    closed_loop = ClosedLoop(LinearPendulum(), idad_policy, scale, shift)

    param_prior = MultivariateNormal(
        torch.tensor([14.7, 0.0, 3.0]), torch.diag(torch.tensor([0.1, 0.01, 0.1]))
    )
    init_state = torch.zeros(3)
    nb_runs = 25
    nb_steps = 50
    nb_trajectories = 16
    nb_particles = 1024

    eig_estimates = torch.zeros(nb_runs)
    for i in range(nb_runs):
        estimate = estimate_eig(nb_steps, nb_trajectories, nb_particles, param_prior, init_state, closed_loop)
        eig_estimates[i] = estimate
        print(f"Run {i}: {estimate:.4f}")

    mean_estimate = eig_estimates.mean()
    std_estimate = eig_estimates.std()
    print(r"EIG estimate: {:.4f} pm {:.4f}".format(mean_estimate, std_estimate))
