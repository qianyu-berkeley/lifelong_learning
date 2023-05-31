import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow.sklearn
import mlflow
import mlflow.tracking as MlflowClient
from mlflow.models.signature import infer_signature
import mlflow.pyfunc
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tempfile



def log_rf_model(experimentID, run_name, params, X_train, X_test, y_train, y_test):
    with mlflow.start_run(experiment_id=experimentID, run_name=run_name) as run:
        # Create model, train it, and create predictions
        rf = RandomForestRegressor(**params)
        rf.fit(X_train, y_train)
        predictions = rf.predict(X_test)

        # Log model
        # with signiture and example
        mlflow.sklearn.log_model(rf, 
                                 "random-forest-model", 
                                 signature=infer_signature(X_train, y_train), 
                                 input_example=X_train.head(3))

        # Log params
        mlflow.log_params(params)

        # Create metrics
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        # Log metrics
        mlflow.log_metrics({"mse": mse, "mae": mae, "r2": r2})

        # Create feature importance
        importance = pd.DataFrame(list(zip(df.columns, rf.feature_importances_)),
                                    columns=["Feature", "Importance"]
                                ).sort_values("Importance", ascending=False)

        # Log importances using a temporary file
        temp = tempfile.NamedTemporaryFile(prefix="feature-importance-", suffix=".csv")
        temp_name = temp.name
        try:
            importance.to_csv(temp_name, index=False)
            mlflow.log_artifact(temp_name, "feature-importance.csv")
        finally:
            temp.close() # Delete the temp file

        # Create plot
        fig, ax = plt.subplots()

        importance.plot.bar(ax=ax)
        plt.xlabel("Predicted values for Price ($)")
        plt.ylabel("Residual")
        plt.title("Residual Plot")

        # Log residuals using a temporary file
        temp = tempfile.NamedTemporaryFile(prefix="residuals-", suffix=".png")
        temp_name = temp.name
        try:
            fig.savefig(temp_name)
            mlflow.log_artifact(temp_name, "residuals.png")
        finally:
            temp.close() # Delete the temp file

        display(fig)
        return run.info.run_id


def query_past_run()
    client = MlflowClient()
    client.list_artifacts(runID)
    runs = pd.DataFrame([(run.run_id, run.start_time, run.artifact_uri) for run in client.list_run_infos(experimentID)])
    runs.columns = ["run_id", "start_time", "artifact_uri"]
    return runs


def get_metric(runID, metric):
    client = MlflowClient()
    client.get_metric_history(runID, metric)
    return client.get_metric_history(runID, metric)

    
def load_model(runID):
    client = MlflowClient()
    model_uri = "runs:/{}/random-forest-model".format(runID)
    model_details = client.get_run(runID).data
    model_details
    model = mlflow.sklearn.load_model(model_uri=model_uri)
    return model

    
def download_artifacts(runID, artifact_path, dst_path):
    client = MlflowClient()
    client.download_artifacts(runID, artifact_path, dst_path)


def create_experiment(experiment_name):
    client = MlflowClient()
    client.create_experiment(experiment_name)
    return client.get_experiment_by_name(experiment_name).experiment_id


def create_experiment_by_path(experiment_path):
    client = MlflowClient()
    mlflow.set_experiment(experiment_path)
    return client.get_experiment_by_name(experiment_path).experiment_id


# load model with mlflow.pyfunc
def load_model_pyfunc(runID):
    client = MlflowClient()
    model_uri = "runs:/{}/random-forest-model".format(runID)
    model_details = client.get_run(runID).data
    model_details
    model = mlflow.pyfunc.load_model(model_uri=model_uri)
    print("load pyfunc model: {type(model)}")
    return model


def model_register(model, model_name, model_version, description, tags):
    mlflow.register_model(model_uri=model_uri, name=model_name, description=description, tags=tags)
    

def model_stage_transition(model_name, model_version, stage):
    client = MlflowClient()
    client.transition_model_version_stage(name=model_name, version=model_version, stage=stage)


def model_detail(model_name, model_version):
    client = MlflowClient()
    model_version_details = client.get_model_version(name=model_name, version=model_version)
    return model_version_details


def model_delete(model_name, model_version):
    client = MlflowClient()
    client.delete_model_version(name=model_name, version=model_version)

# An simple example of inheriting mlflow.pyfunc.PythonModel
class AddN(mlflow.pyfunc.PythonModel):

    def __init__(self, n):
        self.n = n

    def predict(self, context, model_input):
        return model_input.apply(lambda column: column + self.n)

def run_AddN(save=False, reload=False):
    model_path = f"{workingDir}/add_n_model2"
    add5_model = AddN(n=5)
    if save:
        mlflow.pyfunc.save_model(path=model_path.replace("dbfs:", "/dbfs"), python_model=add5_model)
    if reload:
        loaded_model = mlflow.pyfunc.load_model(model_path)
        return loaded_model.predict(pd.DataFrame([range(10)]).T)