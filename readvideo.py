#!/usr/bin/python3

import numpy as np
import cv2
import math

filename = 'test_countryroad.mp4'

cap = cv2.VideoCapture('./videos/'+filename)

frameLength = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

cv2.namedWindow('SLAMView',cv2.WINDOW_NORMAL)
cv2.resizeWindow('SLAMView',600,600)

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

    pts = cv2.goodFeaturesToTrack(np.mean(frame, axis=2).astype(np.uint8), 3000, qualityLevel=0.01, minDistance=5) #mindist 7 used

    # extraction
    kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], _size=20) for f in pts]
    kp, des = orb.compute(frame, kps)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

    matches = bf.match(des,lastDes)

    matches = sorted(matches, key = lambda x:x.distance)

    pointlocs = [k.pt for k in kp]

    # output
    for point in pointlocs:
        cv2.circle(frame,(int(point[0]),int(point[1])),10,(0,0,255),1)

    goodMatchSeq = 0
    matchSeq = 0
    for match in matches:
        matchSeq += 1
        pt1 = kp[match.queryIdx].pt
        pt1 = (int(pt1[0]),int(pt1[1]))
        pt2 = lastKp[match.trainIdx].pt
        pt2 = (int(pt2[0]),int(pt2[1]))

        dist = math.sqrt( (pt1[0]-pt2[0])**2.0 + (pt1[1]-pt2[1])**2.0)
        #if match.distance > 30:
        #    break
        #if dist > 50:
        #    break

        if dist < 100:
            if match.distance < 30:
                goodMatchSeq += 1
                cv2.line(frame,pt1,pt2,(0,255,0),2)

    # frame number and process stats to user
    cv2.putText(frame, filename, (30,30), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0),2)
    printString = str(seq) + "/" + str(frameLength) + "   " + "{0:.2f}".format(100*seq/frameLength) + "%"
    cv2.putText(frame, printString, (30,60), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0),2)
    printString = str(goodMatchSeq) + "/" + str(matchSeq) + "   " + "{0:.2f}".format(100*goodMatchSeq/matchSeq) + "%" + " match pass rate"
    cv2.putText(frame, printString, (30,90), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0),2)
        #cv2.line(frame,pt1,pt2,(0,255,0))

    #
    # pts1 = []
    # pts2 = []
    # good = []
    #
    # # lowe's method
    # for i,(m,n) in enumerate(matches):
    #     if m.distance < 0.8*n.distance:
    #         good.append(m)
    #         pts1.append(kp[m.queryIdx].pt)
    #         pts2.append(lastKp[m.trainIdx].pt)
    #
    # pts1 = np.int32(pts1)
    # pts2 = np.int32(pts2)
    #
    # F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)
    #
    # print(F)

    cv2.imshow('SLAMView',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    lastKp = kp
    lastDes = des


cap.release()
cv2.destroyAllWindows()
