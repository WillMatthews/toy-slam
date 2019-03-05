#!/usr/bin/python3

import numpy as np
import cv2
import math

# TODO:
# add a SIFT detector (maybe LIFT?)
# develop a better match prune method
# non brute force methods
# COMMENTS!
# add a kalman filter to estimate state!
# add a camera calibration estimator
# add a fundamental matrix to translation+rotation estimator
# add a good class structure to this mess somehow

## add a plant model!
#    [x]   [
# d  [y] = [ to be determined
# dt [z]   [

#def is_odd(a):
#    return (a & 1) == 1

filename = 'test_countryroad.mp4'

cap = cv2.VideoCapture('./videos/'+filename)

frameLength = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
orb = cv2.ORB_create()

seq = 1
tickTock = True

ret,frame = cap.read()
lastKp, lastDes = orb.detectAndCompute(frame,None)
height, width, _ = frame.shape

cv2.namedWindow('SLAMView',cv2.WINDOW_NORMAL)
cv2.resizeWindow('SLAMView',600,600)

while(cap.isOpened()):
    seq += 1
    #print(seq)
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
    pts1 = []
    pts2 = []
    for match in matches:
        matchSeq += 1

        # Process a single matched point pair
        pt1 = kp[match.queryIdx].pt
        pt1 = (int(pt1[0]),int(pt1[1]))

        pt2 = lastKp[match.trainIdx].pt
        pt2 = (int(pt2[0]),int(pt2[1]))

        dist = math.sqrt((pt1[0]-pt2[0])**2.0 + (pt1[1]-pt2[1])**2.0)

        if dist < 50:
            if match.distance < 30:
                goodMatchSeq += 1

                #queryPoint = (width//2 - 2*(pt2[0]-width//2),height//2 - 2*(pt2[1]-height//2))
                mF = 20
                #farPoint = (pt2[0] + mF*(pt2[0]-pt1[0]), pt2[1] + mF*(pt2[1]-pt1[1]))

                #cv2.line(frame,pt1,queryPoint,(0,255,0),1)
                #cv2.line(frame,pt1,farPoint,(0,255,0),1)
                cv2.line(frame,pt1,pt2,(128,255,0),2)

                # if the match is good - save it to the list for fundamental matrix calc
                pts1.append(pt1)
                pts2.append(pt2)

    if goodMatchSeq < 10:
        print("BAD MATCHES, this result may be unstable")
        for match in matches:
            # Process a single matched point pair
            pt1 = kp[match.queryIdx].pt
            pt1 = (int(pt1[0]),int(pt1[1]))

            pt2 = lastKp[match.trainIdx].pt
            pt2 = (int(pt2[0]),int(pt2[1]))

            # if the match is good - save it to the list for fundamental matrix calc
            pts1.append(pt1)
            pts2.append(pt2)


    # frame number and process stats to user
    printString = filename + "   " + str(width) + "x" + str(height)
    cv2.putText(frame, printString, (30,30), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0),2)

    printString = str(seq) + "/" + str(frameLength)
    pctString = "{0:.2f}".format(100*seq/frameLength) + "%" + " frame"
    cv2.putText(frame, printString, (30,60), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0),2)
    cv2.putText(frame, pctString, (230,60), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0),2)

    printString = str(goodMatchSeq) + "/" + str(matchSeq)
    pctString = "{0:.2f}".format(100*goodMatchSeq/matchSeq) + "%" + " match pass rate"
    cv2.putText(frame, printString, (30,90), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0),2)
    cv2.putText(frame, pctString, (230,90), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0),2)

    col1 = (0,0,0)
    col2 = (255,255,255)
    if tickTock:
        tickTock = False
    else:
        col1, col2 = col2, col1
        tickTock = True

    cv2.rectangle(frame, (width-20,0), (width,20), col1, -1)
    cv2.rectangle(frame, (width-40,0), (width-20,20), col2, -1)

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
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    print("FUNDAMENTAL MATRIX")
    #
    F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)
    #
    print(F)

    cv2.imshow('SLAMView',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    lastKp = kp
    lastDes = des


cap.release()
cv2.destroyAllWindows()
