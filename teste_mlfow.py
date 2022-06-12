from kaggle.census_to_random_florest import floresta
import mlflow

#save the model
mlflow.sklearn.log_model(floresta, "models")
print(f"Model saved in {mlflow.active_run().info.artifact_uri}")