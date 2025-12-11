import time
import joblib
from HumanPoseEstimation import singleUnitUBHPE
import numpy as np
import cv2

vid = cv2.VideoCapture(1)
estInterval = 0.5
curEst = ''
font = cv2.FONT_HERSHEY_SIMPLEX

model = joblib.load("YMCA-Pose-Estimation/clf-model.joblib")
startTime = time.time()
while True:
    curTime = time.time()
    success, img = vid.read()
    img = cv2.flip(img, 1)
    height, width, channel = img.shape

    if curTime - startTime > estInterval:
        startTime = curTime
        res_lm, img = singleUnitUBHPE(img, False)
        X = np.array([res_lm], dtype=np.float32)
        curEst = str(model.predict(X)[0])

    cv2.rectangle(img, (width//2 - 30, height - 150), (width//2 + 100, height - 30),
                  (255, 255, 255), -1)
    cv2.putText(img, curEst, (width//2, height - 50), font,4,
                (0, 0, 255), 10, cv2.LINE_AA)
    cv2.imshow("Capture", img)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

