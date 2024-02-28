import os
import argparse

import pandas as pd

import torch
import pyro

import mlflow

from experiment_tools.pyro_tools import auto_seed
from experiment_tools.output_utils import get_mlflow_meta
from estimators.mi import PriorContrastiveEstimation


def evaluate_run(
    experiment_id,
    run_id,
    num_experiments_to_perform,
    num_inner_samples,
    num_outer_samples,
    num_seeds,
    device,
    seed,
    # if checkpoints were stored (as model_postfix), pass here
    model_postfix="",
):
    pyro.clear_param_store()
    artifact_path = f"mlruns/{experiment_id}/{run_id}/artifacts"
    model_location = f"{artifact_path}/model{model_postfix}"
    seed = auto_seed(seed)

    EIGs_mean = pd.DataFrame(columns=["lower"])
    EIGs_se = pd.DataFrame(columns=["lower"])

    for t_exp in num_experiments_to_perform:
        # load model, set number of experiments
        trained_model = mlflow.pytorch.load_model(model_location, map_location=device)
        if t_exp:
            trained_model.T = t_exp
        else:
            t_exp = trained_model.T

        pce_loss_lower = PriorContrastiveEstimation(
            trained_model.model, num_outer_samples, num_inner_samples
        )

        auto_seed(seed)
        EIG_proxy_lower = torch.tensor(
            [-pce_loss_lower.loss() for _ in range(num_seeds)]
        )

        EIGs_mean.loc[t_exp, "lower"] = EIG_proxy_lower.mean().item()
        EIGs_se.loc[t_exp, "lower"] = EIG_proxy_lower.std().item()

    EIGs_mean["stat"] = "mean"
    EIGs_se["stat"] = "se"
    res = pd.concat([EIGs_mean, EIGs_se])
    print(res)
    if not os.path.exists("mlflow_outputs"):
        os.makedirs("mlflow_outputs")
    res.to_csv(f"mlflow_outputs/eval{model_postfix}.csv")

    with mlflow.start_run(run_id=run_id, experiment_id=experiment_id) as run:
        mlflow.log_artifact(
            f"mlflow_outputs/eval{model_postfix}.csv", artifact_path="evaluation",
        )
        if len(num_experiments_to_perform) == 1:
            mlflow.log_metric(
                f"eval_mi_lower{model_postfix}", EIGs_mean.loc[t_exp, "lower"],
            )

    return res


def evaluate_experiment(
    experiment_id,
    seed,
    num_experiments_to_perform,
    device,
    num_inner_samples,
    num_outer_samples,
    num_seeds,
    model_postfix="",
):

    filter_string = "params.status='complete'"
    meta = get_mlflow_meta(experiment_id=experiment_id, filter_string=filter_string)
    # run those that haven't yet been evaluated
    meta = [
        m for m in meta if f"eval_mi_lower{model_postfix}" not in m.data.metrics.keys()
    ]
    meta = [m for m in meta if "baseline_type" not in m.data.params.keys()]
    experiment_run_ids = [run.info.run_id for run in meta]
    print(experiment_run_ids)
    for i, run_id in enumerate(experiment_run_ids):
        print(f"Evaluating run {i+1} out of {len(experiment_run_ids)} runs...")
        evaluate_run(
            experiment_id=experiment_id,
            run_id=run_id,
            num_experiments_to_perform=num_experiments_to_perform,
            num_inner_samples=num_inner_samples,
            num_outer_samples=num_outer_samples,
            num_seeds=num_seeds,
            device=device,
            seed=seed,
            model_postfix=model_postfix,
        )
        print("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Deep Adaptive Design: Model Evaluation via sPCE."
    )
    parser.add_argument("--experiment-id", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", default=-1, type=int)
    parser.add_argument("--num-experiments-to-perform", nargs="+", default=[None])
    parser.add_argument("--num_inner_samples", default=1_000_000, type=int)
    parser.add_argument("--num_outer_samples", default=32, type=int)
    parser.add_argument("--num_seeds", default=16, type=int)

    args = parser.parse_args()
    args.num_experiments_to_perform = [
        int(x) if x else x for x in args.num_experiments_to_perform
    ]
    evaluate_experiment(
        experiment_id=args.experiment_id,
        seed=args.seed,
        num_experiments_to_perform=args.num_experiments_to_perform,
        device=args.device,
        num_inner_samples=args.num_inner_samples,
        num_outer_samples=args.num_outer_samples,
        num_seeds=args.num_seeds
    )
