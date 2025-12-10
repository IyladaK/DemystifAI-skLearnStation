from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from sklearn.datasets import load_digits

# --- --- --- THE MODELS --- --- ---
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
# --- --- --- --- --- --- --- --- ---

digits = load_digits()

# !! split the data

# instantiating and fitting the model
myModel = None # instantiate the model here
# fit the model

# predict the test data and check the accuracy (print it)

"""
We use Joblib to pipeline our model.
Joblib is a Python library used for efficiently saving, loading, 
and caching Python objects, especially large machine learning models 
or data.

This way we can avoid retraining our model everytime we want to use it
"""

joblib.dump(myModel, "myModel.joblib")