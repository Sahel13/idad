import torch
import mlflow
from experiment_tools.output_utils import get_mlflow_meta


experiment_id = ""
filter_string = "params.status='complete'"
meta = get_mlflow_meta(experiment_id=experiment_id, filter_string=filter_string)
experiment_run_ids = [run.info.run_id for run in meta]

for run_id in experiment_run_ids:
    artifact_path = f"mlruns/{experiment_id}/{run_id}/artifacts"
    model_location = f"{artifact_path}/model"
    trained_model = mlflow.pytorch.load_model(model_location).model
    # Do something with it.
