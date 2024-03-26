import torch
from eig_estimation.iosmc import IBISDynamics, ClosedLoop, estimate_eig
import argparse
import mlflow
from experiment_tools.output_utils import get_mlflow_meta
from torch.distributions import LogNormal, Independent


torch.manual_seed(123)


class CartPole(IBISDynamics):
    def __init__(self):
        xdim = 4
        udim = 1
        step = 0.05
        diffusion_vector = torch.tensor([0.0, 0.0, 0.1, 0.0])
        super().__init__(xdim, udim, step, diffusion_vector)

    def drift_fn(self, p, x, u):
        l, mp, mc = p
        g = 9.81

        c, k = 1e-2, 1e-2
        d, v = 1e-2, 1e-2

        s, q, ds, dq = x

        sin_q = torch.sin(q)
        cos_q = torch.cos(q)

        dds = (
            u
            - (c * ds + k * s)
            - (d * dq + v * q) * cos_q / l
            + mp * sin_q * (l * dq**2 + g * cos_q)
        ) / (mc + mp * sin_q**2)

        ddq = (
            -u * cos_q
            - mp * l * dq**2 * cos_q * sin_q
            - (mc + mp) * g * sin_q
            - (c * ds + k * s) * cos_q
            - (d * dq + v * q) * cos_q**2 / l
        ) / (l * mc + l * mp * sin_q**2)

        return torch.tensor([ds, dq, dds, ddq])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", type=str, default="694485179914021738")
    args = parser.parse_args()

    # Load the trained policy.
    filter_string = "params.status='complete'"
    meta = get_mlflow_meta(
        experiment_id=args.experiment_id, filter_string=filter_string
    )
    experiment_run_ids = [run.info.run_id for run in meta]
    run_id = experiment_run_ids[0]
    artifact_path = f"mlruns/{args.experiment_id}/{run_id}/artifacts"
    model_location = f"{artifact_path}/model"
    trained_model = mlflow.pytorch.load_model(model_location, map_location="cpu")
    idad_policy = trained_model.design_net

    scale, shift = 5.0, 0.0
    closed_loop = ClosedLoop(CartPole(), idad_policy, scale, shift)

    param_prior = Independent(
        LogNormal(torch.zeros(3), torch.tensor([0.1, 0.1, 0.1])), 1
    )
    init_state = torch.zeros(5)
    nb_runs = 25
    nb_steps = 50
    nb_trajectories = 16
    nb_particles = 1024

    eig_estimates = torch.zeros(nb_runs)
    for i in range(nb_runs):
        estimate = estimate_eig(
            nb_steps,
            nb_trajectories,
            nb_particles,
            param_prior,
            init_state,
            closed_loop,
        )
        eig_estimates[i] = estimate
        print(f"Run {i}: {estimate:.4f}")

    mean_estimate = eig_estimates.mean()
    std_estimate = eig_estimates.std()
    print(r"EIG estimate: {:.2f} pm {:.2f}".format(mean_estimate, std_estimate))
