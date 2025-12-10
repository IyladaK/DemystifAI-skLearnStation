import cv2
import mediapipe as mp

vid = cv2.VideoCapture(1)
mpPose = mp.solutions.pose
pose = mpPose.Pose()

lm_ofInterest = {16 : "L-wrist", 14 : "L-elbow", 12: "L-shoulder",
                     15 : "R-wrist", 13 : "R-elbow", 11: "R-shoulder"}
connections = [(16, 14), (14, 12), (12, 11), (11, 13), (13, 15)]

def singleUnitUBHPE(img, dots : bool = True):
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)

    res_lm = {}
    resArr = []

    # drawing landmarks
    if results.pose_landmarks:
        for id, lm in enumerate(results.pose_landmarks.landmark):
            if id in lm_ofInterest:
                height, width, channel = img.shape
                cx, cy = int(lm.x * width), int(lm.y * height)
                res_lm[id] = (cx, cy)
                resArr.extend([cx, cy])
                if dots:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)

        if dots:
            for conn in connections:
                cv2.line(img=img, pt1=res_lm[conn[0]], pt2=res_lm[conn[1]], color=(255, 255, 255))

    return resArr, img