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
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=0)

# instantiating and fitting the model
myModel = LogisticRegression(max_iter = 10000) # instantiate the model here
# fit the model
myModel.fit(x_train, y_train)

# predict the test data and check the accuracy (print it)
predictions = myModel.predict(x_test)
print(myModel.score(x_test, y_test))

"""
We use Joblib to pipeline our model.
Joblib is a Python library used for efficiently saving, loading, 
and caching Python objects, especially large machine learning models 
or data.

This way we can avoid retraining our model everytime we want to use it
"""

joblib.dump(myModel, "digitDrawing/myModel.joblib")