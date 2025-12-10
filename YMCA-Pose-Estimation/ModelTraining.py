from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import joblib

# Here, we import the csv data that we collected and format them into Dataframes
df = pd.read_csv('pose_data.csv')

Y = df["Label"].values
df = df.drop("Label", axis=1)

X = df.to_numpy()

# !! Split the data into training and testing

# !! Select a model, assing to a variable, then fit the data


# !! predict the testing data and check the accuracy


# !! dump the model
