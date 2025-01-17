if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter

import mlflow
import joblib
@data_exporter
def export_data(args):
    """
    Exports data to some source.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Output (optional):
        Optionally return any object and it'll be logged and
        displayed when inspecting the block run.
    """
    # Specify your data exporting logic here
    mlflow.end_run()
    lr, dv = args
    mlflow.set_experiment(experiment_id="501909255559078474")
    with mlflow.start_run() as run:
        # Log the model
        mlflow.sklearn.log_model(sk_model=lr, artifact_path='model')

        # Log the vectorizer
        joblib.dump(dv, "vectorizer.pkl")
        mlflow.log_artifact("vectorizer.pkl", artifact_path="vectorizer")

    
