# for data manipulation
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)
# for model serialization
import joblib
# for creating a folder
import os
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("mlops-training-experiment")

api = HfApi()


Xtrain_path = "hf://datasets/shashwativatsa/SuperKart/Xtrain.csv"
Xtest_path = "hf://datasets/shashwativatsa/SuperKart/Xtest.csv"
ytrain_path = "hf://datasets/shashwativatsa/SuperKart/ytrain.csv"
ytest_path = "hf://datasets/shashwativatsa/SuperKart/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)


# List of numerical features in the dataset (excluding 'id' as it is an identifier)
numeric_features = [
    'Product_Weight',           # weight of each product
    'Product_Allocated_Area',   # ratio of the allocated display area of each product to the total display area of all the products in a store
    'Product_MRP',              # maximum retail price of each product
    'Store_Establishment_Year', # year in which the store was established
]

# List of categorical features in the dataset
categorical_features = [
    'Product_Sugar_Content',    # sugar content of each product like low sugar, regular and no sugar
    'Product_Type',             # broad category for each product like meat, snack foods etc.
    'Store_Size',               # size of the store depending on sq. feet like high, medium and low
    'Store_Location_City_Type', # type of city in which the store is located like Tier 1, Tier 2 and Tier 3.
    'Store_Type',               # type of store depending on the products that are being sold there like Departmental Store, Food Mart etc.
]


# Define the preprocessing steps
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

# Define base XGBoost model
xgb_model = xgb.XGBRegressor(random_state=42)


# Define hyperparameter grid
param_grid = {
    'xgbregressor__n_estimators': [50, 75, 100],         # number of tree to build
    'xgbregressor__max_depth': [2, 3, 4],                # maximum depth of each tree
    'xgbregressor__colsample_bytree': [0.4, 0.5, 0.6],   # percentage of attributes to be considered (randomly) for each tree
    'xgbregressor__colsample_bylevel': [0.4, 0.5, 0.6],  # percentage of attributes to be considered (randomly) for each level of a tree
    'xgbregressor__learning_rate': [0.01, 0.05, 0.1],    # learning rate
    'xgbregressor__reg_lambda': [0.4, 0.5, 0.6],         # L2 regularization factor
}

# Model pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

# Start MLflow run
with mlflow.start_run():
    # Hyperparameter tuning
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

    # Log all parameter combinations and their mean test scores
    results = grid_search.cv_results_
    for i in range(len(results['params'])):
        param_set = results['params'][i]
        mean_score = results['mean_test_score'][i]
        std_score = results['std_test_score'][i]

        # Log each combination as a separate MLflow run
        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_test_score", mean_score)
            mlflow.log_metric("std_test_score", std_score)

    # Log best parameters separately in main run
    mlflow.log_params(grid_search.best_params_)

    # Store and evaluate the best model
    best_model = grid_search.best_estimator_

    y_pred_train = best_model.predict(Xtrain)
    y_pred_test = best_model.predict(Xtest)

    train_report = {
        "mae": mean_absolute_error(ytrain, y_pred_train),
        "mse": mean_squared_error(ytrain, y_pred_train),
        "rmse": np.sqrt(mean_squared_error(ytrain, y_pred_train)),
        "r2": r2_score(ytrain, y_pred_train),
        "mape": mean_absolute_percentage_error(ytrain, y_pred_train),
    }

    test_report = {
        "mae": mean_absolute_error(ytest, y_pred_test),
        "mse": mean_squared_error(ytest, y_pred_test),
        "rmse": np.sqrt(mean_squared_error(ytest, y_pred_test)),
        "r2": r2_score(ytest, y_pred_test),
        "mape": mean_absolute_percentage_error(ytest, y_pred_test),
    }

    # Log the metrics for the best model
    mlflow.log_metrics({
        "train_mae": train_report['mae'],
        "train_mse": train_report['mse'],
        "train_rmse": train_report['rmse'],
        "train_r2": train_report['r2'],
        "train_mape": train_report['mape'],
        "test_mae": test_report['mae'],
        "test_mse": test_report['mse'],
        "test_rmse": test_report['rmse'],
        "test_r2": test_report['r2'],
        "test_mape": test_report['mape'],
    })

    # Save the model locally
    model_path = "best_prediction_model_v1.joblib"
    joblib.dump(best_model, model_path)

    # Log the model artifact
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved as artifact at: {model_path}")

    # Upload to Hugging Face
    repo_id = "shashwativatsa/prediction-model"
    repo_type = "model"

    # Step 1: Check if the space exists
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Space '{repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Space '{repo_id}' not found. Creating new space...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Space '{repo_id}' created.")

    # create_repo("churn-model", repo_type="model", private=False)
    api.upload_file(
        path_or_fileobj="best_prediction_model_v1.joblib",
        path_in_repo="best_churn_model_v1.joblib",
        repo_id=repo_id,
        repo_type=repo_type,
    )
