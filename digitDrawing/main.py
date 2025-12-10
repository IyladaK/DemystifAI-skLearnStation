import joblib
import drawingGame
import matplotlib.pyplot as plt

drawnArr, out = drawingGame.drawingWindow()

# deserializing our model
model = joblib.load("myModel.joblib")

# !! make the model predict drawnArr

# !! print the prediction

plt.imshow(out, cmap='gray', vmin=0, vmax=16)
plt.show()