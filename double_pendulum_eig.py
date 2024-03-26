import torch
from eig_estimation.iosmc import (
    IBISDynamics,
    ClosedLoop,
    estimate_eig,
    MultivariateLogNormal,
)
import argparse
import mlflow
from experiment_tools.output_utils import get_mlflow_meta


torch.manual_seed(123)


class DoublePendulum(IBISDynamics):
    def __init__(self):
        xdim = 4
        udim = 2
        step = 0.05
        diffusion_vector = torch.tensor([0.0, 0.0, 0.1, 0.1])
        super().__init__(xdim, udim, step, diffusion_vector)

    def drift_fn(self, p, x, u):
        m1, m2, l1, l2 = p
        k1, k2 = 1e-1, 1e-1

        g = 9.81

        q1, q2, dq1, dq2 = x
        u1, u2 = u

        s1, c1 = torch.sin(q1), torch.cos(q1)
        s2, c2 = torch.sin(q2), torch.cos(q2)
        s12 = torch.sin(q1 + q2)

        # Mass
        I_1 = m1 * l1**2
        I_2 = m2 * l2**2

        M_1 = I_1 + I_2 + m2 * l1**2 + 2.0 * m2 * l1 * l2 * c2
        M_2 = I_2 + m2 * l1 * l2 * c2
        M_3 = M_2
        M_4 = I_2

        # Coriolis
        C_1 = 0.0
        C_2 = -m2 * l1 * l2 * (2.0 * dq1 + dq2) * s2
        C_3 = 0.5 * m2 * l1 * l2 * (2.0 * dq1 + dq2) * s2
        C_4 = -0.5 * m2 * l1 * l2 * dq1 * s2

        # Gravity
        tau_1 = -g * ((m1 + m2) * l1 * s1 + m2 * l2 * s12)
        tau_2 = -g * m2 * l2 * s12

        u1 = u1 - k1 * dq1
        u2 = u2 - k2 * dq2

        ddq2 = tau_2 + u2 - C_3 * dq1 - C_4 * dq2
        ddq2 -= M_3 / M_1 * (tau_1 + u1 - C_1 * dq1 - C_2 * dq2)
        ddq2 /= M_4 - M_3 * M_2 / M_1

        ddq1 = tau_1 + u1 - C_1 * dq1 - C_2 * dq2 - M_2 * ddq2
        ddq1 /= M_1

        return torch.tensor([dq1, dq2, ddq1, ddq2])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", type=str, default="607477153518912122")
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

    scale, shift = torch.tensor([4.0, 2.0]), torch.zeros(2)
    closed_loop = ClosedLoop(DoublePendulum(), idad_policy, scale, shift)

    param_prior = MultivariateLogNormal(
        torch.zeros(4), torch.diag(torch.tensor([0.01, 0.01, 0.01, 0.01]))
    )
    init_state = torch.zeros(6)
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
    print(r"EIG estimate: {:.4f} pm {:.4f}".format(mean_estimate, std_estimate))
