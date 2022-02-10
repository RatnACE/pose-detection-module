import mediapipe as mp
import cv2
import time


class poseDetector():
    def __init__(self, mode=False, complex = 1,
                 smooth=True,segment = False,
                 smsegment = True, detectionCon = 0.5, trackCon = 0.5):
        self.mode = mode
        self.complex = complex
        self.smooth = smooth
        self.segment = segment
        self.smsegment = smsegment
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.complex, self.smooth, self.segment,
                                     self.smsegment, self.detectionCon,
                                     self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findPose(self, img, draw=True ):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


        self.rio = self.pose.process(imgRGB)
        # print(rio.pose_landmarks)


        if self.rio.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img,self.rio.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        return img
    def findPosition(self ,img,draw=True):
        lmlist=[]
        if self.rio.pose_landmarks:
            for id, lm in enumerate(self.rio.pose_landmarks.landmark):
                h, w, c = img.shape
                #print(id, lm)
                cx, cy =int(lm.x*w), int(lm.y*h)
                lmlist.append([id,cx,cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (0, 0, 255), cv2.FILLED)
        return lmlist


def main():
    cap = cv2.VideoCapture(0)
    pt = 0
    detector = poseDetector()

    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        lmlist=detector.findPosition(img)
        print(lmlist[14])

        ct = time.time()
        fps = 1 / (ct - pt)
        pt = ct
        # print(fps)
        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 100, 0), 3)
        cv2.imshow("imagine", img)

        cv2.waitKey(1)

if __name__ =="__main__":
    main()