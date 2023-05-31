import click
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

@click.command()
@click.option("--data_path", default="/dbfs/mnt/training/airbnb/sf-listings/airbnb-cleaned-mlflow.csv", type=str)
@click.option("--n_estimators", default=10, type=int)
@click.option("--max_depth", default=20, type=int)
@click.option("--max_features", default="auto", type=str)
def mlflow_rf(data_path, n_estimators, max_depth, max_features):

    with mlflow.start_run() as run:
        # Import the data
        df = pd.read_csv(data_path)
        X_train, X_test, y_train, y_test = train_test_split(df.drop(["price"], axis=1), df[["price"]].values.ravel(), random_state=42)
        
        # Create model, train it, and create predictions
        rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features)
        rf.fit(X_train, y_train)
        predictions = rf.predict(X_test)

        # Log model
        mlflow.sklearn.log_model(rf, "random-forest-model")
        
        # Log params
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("max_features", max_features)

        # Log metrics
        mlflow.log_metric("mse", mean_squared_error(y_test, predictions))
        mlflow.log_metric("mae", mean_absolute_error(y_test, predictions))  
        mlflow.log_metric("r2", r2_score(y_test, predictions))  


if __name__ == "__main__":
    # Note that this does not need arguments thanks to click
    mlflow_rf() 