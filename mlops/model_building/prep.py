# for data manipulation
import pandas as pd
import sklearn
import os                                             # for creating a folder
from sklearn.model_selection import train_test_split  # for data preprocessing and pipeline creation
from huggingface_hub import login, HfApi              # for hugging face space authentication to upload files

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/shashwativatsa/SuperKart/SuperKart.csv"
superkart_dataset = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Define the target variable for the regression task
target = 'Product_Store_Sales_Total'

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

# Define predictor matrix (X) using selected numeric and categorical features
X = superkart_dataset[numeric_features + categorical_features]

# Define target variable
y = superkart_dataset[target]


# Split the dataset into training and test sets
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y,              # Predictors (X) and target variable (y)
    test_size=0.2,     # 20% of the data is reserved for testing
    random_state=42    # Ensures reproducibility by setting a fixed random seed
)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)


files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="shashwativatsa/SuperKart ",
        repo_type="dataset",
    )
