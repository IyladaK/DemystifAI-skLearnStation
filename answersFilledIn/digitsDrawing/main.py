import joblib
import drawingGame
import matplotlib.pyplot as plt

drawnArr, out = drawingGame.drawingWindow()

# deserializing our model
model = joblib.load("digitDrawing/myModel.joblib")

# !! make the model predict drawnArr
prediction = model.predict(drawnArr)

# !! print the prediction
print(prediction[0])

plt.imshow(out, cmap='gray', vmin=0, vmax=16)
plt.show()