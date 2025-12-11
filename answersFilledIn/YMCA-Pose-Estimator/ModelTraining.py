from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import joblib

df = pd.read_csv('YMCA-Pose-Estimation/pose_data.csv')

Y = df["Label"].values
df = df.drop("Label", axis=1)

X = df.to_numpy()

xTrain, xTest, yTrain, yTest = train_test_split(X, Y, train_size=0.8, random_state=1)
clf = RandomForestClassifier(n_estimators=100)
clf.fit(xTrain, yTrain)

prediction = clf.predict(xTest)
print("Accuracy:", accuracy_score(yTest, prediction))

joblib.dump(clf, "YMCA-Pose-Estimation/clf-model.joblib")