import argparse
import os

import mlflow
import mlflow.pytorch
import pandas as pd
import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn
from pyro.infer.util import torch_item
from tqdm import trange

from estimators.mi import PriorContrastiveEstimation
from experiment_tools.pyro_tools import auto_seed
from neural.aggregators import LSTMImplicitDAD
from neural.modules import Mlp
from oed.design import OED
from oed.primitives import compute_design, latent_sample, observation_sample


class LinearPendulum(nn.Module):
    def __init__(self, design_net, device, T):
        super(LinearPendulum, self).__init__()
        self.design_net = design_net
        self.T = T
        self.dt = 0.05
        self.scale = 1.0
        self.shift = 0.0
        self.theta_prior = dist.MultivariateNormal(
            torch.tensor([14.7, 0.0, 3.0], device=device),
            torch.diag(torch.tensor([0.1, 0.01, 0.1], device=device)),
        )
        self.init_state = torch.tensor([0.0, 0.0], device=device)
        self.diffusion_vector = torch.tensor([0.0, 1e-1], device=device)
        self.cov = torch.diag(self.diffusion_vector**2 * self.dt + 1e-8)

    def ode(self, x, u, theta):
        p1, p2, p3 = [theta[..., [i]] for i in range(3)]
        q, dq = x[..., [0]], x[..., [1]]
        ddq = -torch.sin(q) * p1 - dq * p2 + u * p3
        if dq.size() != ddq.size():
            # Add dimensions to dq
            dq = dq.expand(ddq.size())
        return torch.cat([dq, ddq], dim=-1)

    def _simulator(self, x, u, theta):
        euler_mean = x + self.dt * self.ode(x, u, theta)
        return dist.MultivariateNormal(euler_mean, scale_tril=torch.sqrt(self.cov))

    def _transform_design(self, xi_untransformed):
        return self.shift + self.scale * nn.Tanh()(xi_untransformed)

    def model(self):
        # Not sure what this does.
        if hasattr(self.design_net, "parameters"):
            pyro.module("design_net", self.design_net)

        ################################################################################
        # Sample theta
        ################################################################################
        # sample log-theta and exponentiate
        theta = latent_sample("theta", self.theta_prior)

        y_outcomes = []
        xi_designs = []
        y = self.init_state

        for t in range(self.T):
            ####################################################################
            # Get a design xi
            ####################################################################
            xi_untransformed = compute_design(
                f"xi{t + 1}", self.design_net.lazy(*zip(xi_designs, y_outcomes))
            )
            xi = self._transform_design(xi_untransformed)

            ####################################################################
            # Sample y
            ####################################################################
            _sim = self._simulator(x=y, u=xi, theta=theta)
            y = observation_sample(f"y{t + 1}", _sim)
            # y = observation_sample(f"y{t + 1}", self.simulator, xi, theta)

            ####################################################################
            # Update history
            ####################################################################
            y_outcomes.append(y)
            xi_designs.append(xi_untransformed)  #! work with untransformed designs

        # T-steps experiment
        return xi_designs, y_outcomes, theta

    def forward(self, theta):
        """Run the policy"""
        self.design_net.eval()

        def model():
            with pyro.plate_stack("expand_theta_test", [theta.shape[0]]):
                # condition on theta
                return pyro.condition(self.model, data={"theta": theta})()

        with torch.no_grad():
            designs, outcomes, theta = model()

        self.design_net.train()
        return designs, outcomes

    def eval(self, n_trace=3, theta=None, verbose=True):
        self.design_net.eval()

        if theta is None:
            theta = self.theta_prior.sample(torch.Size([n_trace]))
        else:
            theta = theta.unsqueeze(0).expand(n_trace, *theta.shape)
            # dims: [n_trace * number of thetas given, shape of theta]
            theta = theta.reshape(
                -1,
                *theta.shape[2:],
            )

        output = []
        designs, outcomes = self.forward(theta)

        for i in range(n_trace):
            run_xis = []
            run_ys = []

            if verbose:
                print("Example run {}".format(i))
                print(f"*True Theta: {theta[i].cpu()}*")

            for t in range(self.T):
                xi_untransformed = designs[t][i].detach()
                xi = self._transform_design(xi_untransformed).cpu().reshape(-1)
                run_xis.append(xi)

                y = outcomes[t][i].detach().cpu().reshape(-1)
                run_ys.append(y)
                if verbose:
                    print(f"xi{t + 1}: {run_xis[-1]},  y{t + 1}: {run_ys[-1]}")

            run_df = pd.DataFrame(torch.stack(run_xis).numpy())
            run_df.columns = [f"xi_{i}" for i in range(xi.shape[0])]
            run_df["observations"] = run_ys
            run_df["order"] = list(range(1, self.T + 1))
            run_df["run_id"] = i + 1
            output.append(run_df)

        self.design_net.train()
        return pd.concat(output), theta.cpu().numpy()


def train_model(
    num_steps,
    batch_size,
    num_negative_samples,
    seed,
    lr,
    gamma,
    device,
    T,
    hidden_dim,
    encoding_dim,
    mlflow_experiment_name,
):
    pyro.clear_param_store()

    ### Set up Mlflow logging ### ------------------------------------------------------
    mlflow.set_experiment(mlflow_experiment_name)
    seed = auto_seed(seed)

    #####
    n = 2  # Output dim
    design_dim = 1
    observation_dim = n

    mlflow.log_param("seed", seed)
    mlflow.log_param("num_experiments", T)
    mlflow.log_param("lr", lr)
    mlflow.log_param("gamma", gamma)
    mlflow.log_param("num_steps", num_steps)
    mlflow.log_param("hidden_dim", hidden_dim)
    mlflow.log_param("encoding_dim", encoding_dim)
    # ----------------------------------------------------------------------------------

    ###################################################################################
    ### Setup design and critic networks
    ###################################################################################
    ### DESIGN NETWORK ###
    history_encoder = Mlp(
        input_dim=[design_dim, observation_dim],
        hidden_dim=[hidden_dim, hidden_dim],  # hidden_dim,
        output_dim=encoding_dim,
        activation=nn.ReLU(),
        name="policy_history_encoder",
    )
    design_emitter = Mlp(
        # iDAD only -> options are sum or cat
        input_dim=encoding_dim,
        hidden_dim=[hidden_dim, hidden_dim],
        output_dim=design_dim,
        activation=nn.ReLU(),
        name="policy_design_emitter",
    )
    # iDAD LSTM aggregator
    design_net = LSTMImplicitDAD(
        history_encoder,
        design_emitter,
        empty_value=torch.zeros(design_dim, device=device),
        num_hidden_layers=2,
    ).to(device)

    #######################################################################
    # print("initilize net")
    # design_net.apply(init_weights)
    pendulum = LinearPendulum(design_net, device, T)

    def separate_learning_rate(module_name, param_name):
        if module_name == "critic_net":
            return {"lr": lr}
        elif module_name == "design_net":
            return {"lr": lr}
        else:
            raise NotImplementedError()

    optimizer = torch.optim.Adam
    patience = 1
    annealing_freq = 400
    mlflow.log_param("annealing_scheme", [annealing_freq, patience, gamma])
    scheduler = pyro.optim.ReduceLROnPlateau(
        {
            "optimizer": optimizer,
            "optim_args": separate_learning_rate,
            "factor": gamma,
            "patience": patience,
            "verbose": False,
        }
    )

    mi_loss_instance = PriorContrastiveEstimation(
        model=pendulum.model,
        batch_size=batch_size,
        num_negative_samples=num_negative_samples,
    )

    mlflow.log_param("num_negative_samples", mi_loss_instance.num_negative_samples)
    mlflow.log_param("num_batch_samples", mi_loss_instance.batch_size)

    oed = OED(optim=scheduler, loss=mi_loss_instance)

    loss_history = []
    outputs_history = []

    # num_steps_range = trange(1, num_steps + 1, desc="Loss: 0.000 ")
    num_steps_range = trange(0, num_steps + 0, desc="Loss: 0.000 ")
    # Evaluate model and store designs for this latent:
    test_theta = torch.tensor([10.0, 0.0, 5.0], device=device).unsqueeze(0)

    ### Log params:
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    mlflow.log_param("num_params_designnet", count_parameters(design_net))

    for i in num_steps_range:
        pendulum.train()
        loss = oed.step()
        loss = torch_item(loss)
        loss_history.append(loss)
        num_steps_range.set_description("Loss: {:.3f} ".format(loss))

        if i % annealing_freq == 0:
            # Log the loss
            loss_eval = oed.evaluate_loss()
            mlflow.log_metric("loss", loss_eval, step=i)
            # Check if lr should be decreased.
            scheduler.step(loss_eval)
            df, latents = pendulum.eval(n_trace=1, theta=test_theta, verbose=False)
            df["step"] = i
            outputs_history.append(df)

    if not os.path.exists("mlflow_outputs"):
        os.makedirs("mlflow_outputs")

    pd.concat(outputs_history).reset_index().to_csv(f"mlflow_outputs/designs_hist.csv")
    mlflow.log_artifact(f"mlflow_outputs/designs_hist.csv", artifact_path="designs")

    pendulum.eval()
    # store params, metrics and artifacts to mlflow ------------------------------------
    print("Storing model to MlFlow... ", end="")
    mlflow.pytorch.log_model(pendulum.cpu(), "model")

    ml_info = mlflow.active_run().info
    model_loc = f"mlruns/{ml_info.experiment_id}/{ml_info.run_id}/artifacts"
    print(f"Model and critic sotred in {model_loc}. Done.")
    mlflow.log_param("status", "complete")

    return pendulum


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="iDAD: Conditionally-Linear Pendulum")
    parser.add_argument("--num-steps", default=10000, type=int)
    parser.add_argument("--num-batch-samples", default=512, type=int)
    parser.add_argument("--num-negative-samples", default=16383, type=int)
    parser.add_argument("--seed", default=-1, type=int)
    parser.add_argument("--lr", default=0.0005, type=float)
    parser.add_argument("--gamma", default=0.96, type=float)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--num-experiments", default=50, type=int)
    parser.add_argument("--hidden-dim", default=256, type=int)
    parser.add_argument("--encoding-dim", default=64, type=int)
    parser.add_argument("--mlflow-experiment-name", default="pendulum_linear", type=str)
    args = parser.parse_args()

    train_model(
        num_steps=args.num_steps,
        batch_size=args.num_batch_samples,
        num_negative_samples=args.num_negative_samples,
        seed=args.seed,
        lr=args.lr,
        gamma=args.gamma,
        T=args.num_experiments,
        device=args.device,
        hidden_dim=args.hidden_dim,
        encoding_dim=args.encoding_dim,
        mlflow_experiment_name=args.mlflow_experiment_name,
    )
