import cv2
import time
import posemodule as pm
cap = cv2.VideoCapture(0)
pt = 0
detector = pm.poseDetector()

while True:
    success, img = cap.read()
    img = detector.findPose(img)
    lmlist=detector.findPosition(img)
    #print(lmlist[24])

    ct = time.time()
    fps = 1 / (ct - pt)
    pt = ct
    # print(fps)
    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 100, 0), 3)
    cv2.imshow("imagine", img)

    cv2.waitKey(1)