#!/usr/bin/python3

import numpy as np
import cv2

filename = 'test_countryroad.mp4'

cap = cv2.VideoCapture('./videos/'+filename)

cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
cv2.resizeWindow('frame',600,600)

orb = cv2.ORB_create()

seq = 0

ret,frame = cap.read()
lastKp, lastDes = orb.detectAndCompute(frame,None)

while(cap.isOpened()):
    seq += 1
    print(seq)
    ret, frame = cap.read()
    if not ret:
        break

    pts = cv2.goodFeaturesToTrack(np.mean(frame, axis=2).astype(np.uint8), 3000, qualityLevel=0.01, minDistance=7)

    # extraction
    kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], _size=20) for f in pts]
    kp, des = orb.compute(frame, kps)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

    matches = bf.match(des,lastDes)

    matches = sorted(matches, key = lambda x:x.distance)

    pointlocs = [k.pt for k in kp]

    for point in pointlocs:
        cv2.circle(frame,(int(point[0]),int(point[1])),10,(0,0,255),1)


    cv2.imshow('frame',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    lastKp = kp
    lastDes = des


cap.release()
cv2.destroyAllWindows()

