import mediapipe as mp
import cv2
import time

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)
pt = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rio = pose.process(imgRGB)
    #print(rio.pose_landmarks)
    if rio.pose_landmarks:
        mpDraw.draw_landmarks(img,rio.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(rio.pose_landmarks.landmark):
            h, w, c = img.shape
            print(id, lm)
            cx, cy =int(lm.x*w), int(lm.y*h)
            cv2.circle(img, (cx, cy), 5, (0, 0, 255), cv2.FILLED)
    ct = time.time()
    fps = 1/(ct-pt)
    pt = ct
    #print(fps)
    cv2.putText(img, str(int(fps)), (70,50), cv2.FONT_HERSHEY_PLAIN, 3, (0,100,0), 3)
    cv2.imshow("imagine", img)


    cv2.waitKey(1)