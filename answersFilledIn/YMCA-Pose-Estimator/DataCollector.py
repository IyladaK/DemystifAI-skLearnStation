from HumanPoseEstimation import singleUnitUBHPE
import cv2
import time
import pandas as pd

"""
--- --- --- USER VARIABLES --- --- ---
"""
LABEL = 'A' # <-- Change this to adjust the current data collection letter
DURATION = 10 # <-- Change this to adjust how long you're collecting for
INTERVAL = 0.25 # <-- Change this to adjust how often datapoints are taken


"""
--- --- --- --- PROCESS--- --- --- ---
"""
bodyParts = ["R-shoulder_x", "R-shoulder_y", "L-shoulder_x", "L-shoulder_y",
             "R-elbow_x", "R-elbow_y", "L-elbow_x", "L-elbow_y",
             "R-wrist_x", "R-wrist_y", "L-wrist_x", "L-wrist_y",]
collected_data = []

# cv2 setup variables
cap = cv2.VideoCapture(1)
font = cv2.FONT_HERSHEY_SIMPLEX


print("GET READY")
for i in range(3, 0, -1):
    print(i)
    time.sleep(1)

startTime = time.time()
prevTime = time.time()

while time.time() - startTime < DURATION:
    # video capture setup
    success, img = cap.read()
    height, width, channel = img.shape
    img = cv2.flip(img, 1)

    # conducting the upper body HPE
    resArr, img = singleUnitUBHPE(img)

    curTime = time.time()
    elapsed = int(curTime - startTime)

    # initial delay interval longer
    if elapsed < 3.0:
        cv2.putText(img, str(int(time.time() - startTime)), (10, 100), font,
                    4, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(img, LABEL, (width - 100, 100), font,
                    4, (0, 0, 255), 2, cv2.LINE_AA)
    # count up to duration
    else:
        cv2.putText(img, str(int(time.time() - startTime)), (10, 100), font,
                    4, (255, 255, 255), 2, cv2.LINE_AA)

    # initial delay interval is longer
    if elapsed < 3.0:
        pass
    # after initial delay, take data points at every collection interval
    elif curTime - prevTime >= INTERVAL:
        cv2.rectangle(img, (0, 0), (width, height), (255, 255, 255), 40)
        collected_data.append(resArr)
        prevTime = curTime

    cv2.imshow("Capture", img)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

# prints for checking purposes
for arr in collected_data:
    print(arr)

# making panda dataframe from the collected data
df = pd.DataFrame(collected_data, columns=bodyParts)
df['Label'] = LABEL # <-- adding a label column with the current collection letter

# concatenating old and new dataframes
db = pd.read_csv("YMCA-Pose-Estimation/pose_data.csv")
add = input("add to database?")

# if the data is not sound, you can choose to not add it
if add == 'y':
    pd.concat([db, df], ignore_index=True).to_csv("YMCA-Pose-Estimation/pose_data.csv", index=False)


