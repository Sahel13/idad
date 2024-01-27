import os
import pickle
import argparse
import math

import pandas as pd
from tqdm import trange
import torch
import torch.nn as nn
import pyro
from pyro.infer.util import torch_item
import pyro.distributions as dist
import mlflow
import mlflow.pytorch

from oed.primitives import observation_sample, latent_sample, compute_design
from experiment_tools.pyro_tools import auto_seed
from oed.design import OED
from estimators.bb_mi import InfoNCE, NWJ
from estimators.mi import PriorContrastiveEstimation


from neural.modules import Mlp
from neural.aggregators import (
    PermutationInvariantImplicitDAD,
    LSTMImplicitDAD,
    ConcatImplicitDAD,
)
from neural.baselines import (
    ConstantBatchBaseline,
    BatchDesignBaseline,
    RandomDesignBaseline,
)
from neural.critics import CriticDotProd, CriticBA

mi_estimator_options = {
    "NWJ": NWJ,
    "InfoNCE": InfoNCE,
    "sPCE": PriorContrastiveEstimation
}

class SimplePendulum(nn.Module):
    """
    Class for the SDE version of the simple pendulum.
    Data is generated on the fly.
    """
    def __init__(
            self,
            design_net,
            device,
            T = 25,
            dt = 0.05,
            scale = 2.5,
            shift = 0.0
    ):
        super(SimplePendulum, self).__init__()
        self.p = 2
        self.design_net = design_net
        self.T = T
        self.dt = dt
        self.cov = torch.diag(torch.tensor([0.001, dt], device=device))
        self.scale = scale
        self.shift = shift
        self.log_theta_prior = dist.MultivariateNormal(
            torch.tensor([0.3, 0.3], device=device), torch.diag(torch.tensor([0.4, 0.4], device=device))
        )
        self.init_state = torch.tensor([0.0, 0.0], device=device)

    def ode(self, x, u, theta):
        """
        ODE of the simple pendulum.
        Should be able to deal with any permutation of
        single or batched `x`, `u` and `theta`.
        """
        m, l = [theta[..., [i]] for i in range(2)]
        q, dq = x[..., [0]], x[..., [1]]
        ddq = -9.81 * torch.sin(q) / l + u / (m * l ** 2)
        if len(dq.size()) == 1:
            dq = torch.full(ddq.size(), dq.item(), device=ddq.device)
        return torch.hstack([dq, ddq])

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
        theta = latent_sample("log_theta", self.log_theta_prior).exp()

        y_outcomes = []
        xi_designs = []
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
            if t > 0:
                _sim = self._simulator(x=y_outcomes[-1], u=xi, theta=theta)
            else:
                _sim = self._simulator(x=self.init_state, u=xi, theta=theta)

            y = observation_sample(f"y{t + 1}", _sim)
            # y = observation_sample(f"y{t + 1}", self.simulator, xi, theta)

            ####################################################################
            # Update history
            ####################################################################
            y_outcomes.append(y)
            xi_designs.append(xi_untransformed)  #! work with untransformed designs

        # T-steps experiment
        return xi_designs, y_outcomes, theta

    def forward(self, log_theta):
        """Run the policy"""
        self.design_net.eval()

        def model():
            with pyro.plate_stack("expand_theta_test", [log_theta.shape[0]]):
                # condition on theta
                return pyro.condition(self.model, data={"log_theta": log_theta})()

        with torch.no_grad():
            designs, outcomes, theta = model()

        self.design_net.train()
        return designs, outcomes

    def eval(self, n_trace=3, log_theta=None, verbose=True):
        self.design_net.eval()

        if log_theta is None:
            log_theta = self.log_theta_prior.sample(torch.Size([n_trace]))
        else:
            log_theta = log_theta.unsqueeze(0).expand(n_trace, *log_theta.shape)
            # dims: [n_trace * number of thetas given, shape of theta]
            log_theta = log_theta.reshape(-1, *log_theta.shape[2:],)

        output = []
        designs, outcomes = self.forward(log_theta)
        theta = log_theta.exp()

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
    lr_critic,
    gamma,
    device,
    T,
    hidden_dim,
    encoding_dim,
    critic_arch,
    mi_estimator,
    mlflow_experiment_name,
    design_arch,
):
    pyro.clear_param_store()

    ### Set up Mlflow logging ### ------------------------------------------------------
    mlflow.set_experiment(mlflow_experiment_name)
    seed = auto_seed(seed)

    #####
    n = 2 # Output dim
    design_dim = 1
    latent_dim = 2
    observation_dim = n

    if lr_critic is None:
        lr_critic = lr
    # Design emitter hidden layer
    des_emitter_HD = encoding_dim // 2

    # History encoder is applied to encoding of both design and critic networks.
    hist_encoder_HD = [8, 64, hidden_dim]

    # These are for critic only:
    latent_encoder_HD = [8, 64, hidden_dim]
    hist_enc_critic_head_HD = encoding_dim // 2

    mlflow.log_param("HD_hist_encoder", str(hist_encoder_HD))
    mlflow.log_param("HD_des_emitter", str(des_emitter_HD))
    mlflow.log_param("HD_latent_encoder", str(latent_encoder_HD))
    mlflow.log_param("HD_hist_enc_critic_head", str(hist_enc_critic_head_HD))

    mlflow.log_param("seed", seed)
    mlflow.log_param("num_experiments", T)
    mlflow.log_param("lr", lr)
    mlflow.log_param("lr_critic", lr_critic)
    mlflow.log_param("gamma", gamma)
    mlflow.log_param("num_steps", num_steps)
    mlflow.log_param("hidden_dim", hidden_dim)
    mlflow.log_param("encoding_dim", encoding_dim)

    mlflow.log_param("critic_arch", critic_arch)
    mlflow.log_param("lr_critic", lr_critic)

    mlflow.log_param("design_arch", design_arch)
    mlflow.log_param("mi_estimator", mi_estimator)
    # ----------------------------------------------------------------------------------

    ###################################################################################
    ### Setup design and critic networks
    ###################################################################################
    ### DESIGN NETWORK ###
    history_encoder = Mlp(
        input_dim=[design_dim, observation_dim],
        hidden_dim=hist_encoder_HD,  # hidden_dim,
        output_dim=encoding_dim,
        activation=nn.ReLU(),
        name="policy_history_encoder",
    )
    design_emitter = Mlp(
        # iDAD only -> options are sum or cat
        input_dim=encoding_dim * max((T - 1), 1)
        if design_arch == "cat"
        else encoding_dim,
        hidden_dim=des_emitter_HD,
        output_dim=design_dim,
        activation=nn.ReLU(),
        name="policy_design_emitter",
    )
    if design_arch == "lstm":
        # iDAD LSTM aggregator
        design_net = LSTMImplicitDAD(
            history_encoder,
            design_emitter,
            empty_value=torch.zeros(design_dim, device=device),
            num_hidden_layers=2,
        ).to(device)
    elif design_arch == "random":
        # Random baseline
        # no design net, can be independent or TS
        design_net = RandomDesignBaseline(
            design_dim,
            random_designs_dist=torch.distributions.Uniform(
                torch.tensor(-5.0, device=device), torch.tensor(5.0, device=device)
            ),
        ).to(device)
    elif design_arch == "equal_interval":
        # Equal interval baseline
        linspace = torch.linspace(0.01, 0.99, T, dtype=torch.float32)
        mlflow.log_param("init_design", str(list(linspace.numpy())))
        transformed_designs = linspace.to(device).unsqueeze(1)
        const_designs = torch.log(transformed_designs / (1 - transformed_designs))
        design_net = ConstantBatchBaseline(const_designs=const_designs).to(device)
    elif design_arch == "static":
        # Static baseline
        # can be independent or TS
        design_net = BatchDesignBaseline(
            T=T,
            design_dim=design_dim,
            design_init=torch.distributions.Uniform(
                torch.tensor(-5.0, device=device), torch.tensor(5.0, device=device)
            ),
        )
        mlflow.log_param("init_design", "u(-5, 5)")

    ######## CRITIC NETWORK #######
    ## Latent encoder
    critic_latent_encoder = Mlp(
        input_dim=latent_dim,
        hidden_dim=latent_encoder_HD,
        output_dim=encoding_dim,
        activation=nn.ReLU(),
        name="critic_latent_encoder",
    )
    ## History encoder
    critic_design_obs_encoder = Mlp(
        input_dim=[design_dim, observation_dim],
        hidden_dim=hist_encoder_HD,
        output_dim=encoding_dim,
        name="critic_design_obs_encoder",
    )
    critic_head = Mlp(
        input_dim=encoding_dim * T if critic_arch == "cat" else encoding_dim,
        hidden_dim=hist_enc_critic_head_HD,
        output_dim=encoding_dim,
        activation=nn.ReLU(),
        name="critic_head",
    )

    if critic_arch == "cat":
        critic_history_encoder = ConcatImplicitDAD(
            encoder_network=critic_design_obs_encoder,
            emission_network=critic_head,
            empty_value=torch.ones(n, latent_dim, device=device),
            T=T,
        )
    elif critic_arch == "lstm":
        critic_history_encoder = LSTMImplicitDAD(
            encoder_network=critic_design_obs_encoder,
            emission_network=critic_head,
            empty_value=torch.ones(n, latent_dim, device=device),
            num_hidden_layers=2,
        )
    elif critic_arch == "sum":
        critic_history_encoder = PermutationInvariantImplicitDAD(
            encoder_network=critic_design_obs_encoder,
            emission_network=critic_head,
            empty_value=torch.ones(n, latent_dim, device=device),
        )
    else:
        raise ValueError("Invalid critic_arch")

    critic_net = CriticDotProd(
        history_encoder_network=critic_history_encoder,
        latent_encoder_network=critic_latent_encoder,
    ).to(device)

    #######################################################################
    # print("initilize net")
    # design_net.apply(init_weights)
    pendulum = SimplePendulum(design_net, device, T=T)

    def separate_learning_rate(module_name, param_name):
        if module_name == "critic_net":
            return {"lr": lr_critic}
        elif module_name == "design_net":
            return {"lr": lr}
        else:
            raise NotImplementedError()

    optimizer = torch.optim.Adam
    patience = 5
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

    if mi_estimator == "sPCE":
        mi_loss_instance = PriorContrastiveEstimation(
            model=pendulum.model,
            batch_size=batch_size,
            num_negative_samples=num_negative_samples,
        )
    else:
        mi_loss_instance = mi_estimator_options[mi_estimator](
            model=pendulum.model,
            critic=critic_net,
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
    test_log_theta = torch.tensor([1.0, 1.5], device=device).log().unsqueeze(0)

    ### Log params:
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    mlflow.log_param("num_params_criticnet", count_parameters(critic_net))
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
            df, latents = pendulum.eval(
                n_trace=1, log_theta=test_log_theta, verbose=False
            )
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
    print("Storing critic network to MlFlow... ", end="")
    mlflow.pytorch.log_model(critic_net.cpu(), "critic")

    ml_info = mlflow.active_run().info
    model_loc = f"mlruns/{ml_info.experiment_id}/{ml_info.run_id}/artifacts"
    print(f"Model and critic sotred in {model_loc}. Done.")
    mlflow.log_param("status", "complete")

    return pendulum


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="iDAD: SDE-Based Pendulum Model")
    parser.add_argument("--num-steps", default=100000, type=int)
    parser.add_argument("--num-batch-samples", default=512, type=int)
    parser.add_argument("--num-negative-samples", default=63, type=int)
    parser.add_argument("--seed", default=-1, type=int)
    parser.add_argument("--lr", default=0.0005, type=float)
    parser.add_argument("--lr-critic", default=None, type=float)
    parser.add_argument("--gamma", default=0.96, type=float)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--num-experiments", default=5, type=int)
    parser.add_argument("--hidden-dim", default=256, type=int)
    parser.add_argument("--encoding-dim", default=64, type=int)
    parser.add_argument("--mi-estimator", default="InfoNCE", type=str)
    # cat, lstm (suitable for ts) or sum (suitable for iid)
    parser.add_argument(
        "--critic-arch", default="lstm", type=str, choices=["cat", "sum", "lstm"]
    )
    # iDAD: <sum> or <lstm>
    # Baselines: choice between  <static>, <equal_interval> and <random>
    parser.add_argument(
        "--design-arch",
        default="lstm",
        type=str,
        choices=["sum", "lstm", "static", "equal_interval", "random"],
    )

    parser.add_argument("--mlflow-experiment-name", default="pendulum", type=str)
    args = parser.parse_args()

    train_model(
        num_steps=args.num_steps,
        batch_size=args.num_batch_samples,
        num_negative_samples=args.num_negative_samples,
        seed=args.seed,
        lr=args.lr,
        lr_critic=args.lr_critic,
        gamma=args.gamma,
        device=args.device,
        T=args.num_experiments,
        hidden_dim=args.hidden_dim,
        encoding_dim=args.encoding_dim,
        critic_arch=args.critic_arch,
        mi_estimator=args.mi_estimator,
        mlflow_experiment_name=args.mlflow_experiment_name,
        design_arch=args.design_arch,
    )
