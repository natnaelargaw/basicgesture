from __future__ import division
import cv2
import math
import time
import numpy as np

def trimmer(data1, w, h):

    inputimg =data1.copy()
    left = inputimg[0:w-5, 0:h] #vertically
    right= inputimg[0:5, 0:h]
    veritcal = np.concatenate((right, left), axis=0) #vertically merge

    top = veritcal[0:w, 0:h-5]
    bottom = veritcal[0:w, 0:5]
    final= np.concatenate((bottom, top), axis=1) #vertically merge

    tmpdiff = cv2.absdiff(inputimg, final)  # be frame and ref free
    tmpthresh = cv2.threshold(tmpdiff, 25, 255, cv2.THRESH_BINARY)[1]
    tmpthresh = cv2.erode(tmpthresh, None, iterations=1)
    tmpthresh = cv2.dilate(tmpthresh, None, iterations=5)
    tmpgray = cv2.cvtColor(tmpthresh, cv2.COLOR_BGR2GRAY)


    (conts, _) = cv2.findContours(tmpgray, cv2.RETR_EXTERNAL,
                             cv2.CHAIN_APPROX_SIMPLE)  # discard/add the prefix _, before cnts, if it raise error
    flag = passx = passy = passw =passh =0 # initialize contour coordinates

    for cx in conts:
        (xx2, yy2, ww2, hh2) = cv2.boundingRect(cx)
        if cv2.contourArea(cx) > 1000:
                                    # moments2 = cv2.moments(cx)
                                    # if moments2['m00'] != 0:
                                    #     cx2 = int(moments2['m10'] / moments2['m00'])  # cx = M10/M00
                                    #     cy2 = int(moments2['m01'] / moments2['m00'])  # cy = M01/M00
                                    #Filter between contours arise after identifying the major contour




            if not flag:
                passx=xx2;passy=yy2;passw=ww2; passh=hh2
                flag=1
            else:
                if xx2<=3 or yy2<=3: #smothing shift edges
                    continue
                elif passw < ww2 and passh < hh2:
                        passx=xx2;passy=yy2;passw=ww2;

            if passh >250: passh = int(passh/2) #to make unwanted hand-part discarded
            passw = passw - 5*2 ;   passx= passx+5 ;        passy =passy+5 # to get rid of dilation effects
    print(passh)
    cv2.rectangle(inputimg, (passx, passy), (passx + passw, passy + passh), (255, 0, 255),1)

    secondpass= inputimg[passy:passy+passh, passx:passx+passw]

    cv2.imshow("Sample Points", secondpass)













    cv2.imshow("Local Output", inputimg)
    return [passx,passy, passw, passh]


def entry():
    ref = cv2.imread("D1.jpg")
    frame = cv2.imread("D2.jpg")
    diff1 = cv2.absdiff(ref, frame)  # be frame and ref free
    morphD=15
    thresh = cv2.threshold(diff1, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=1)
    thresh = cv2.dilate(thresh, None, iterations=morphD)
    gray = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Sample",thresh)
    (cnts, _) = cv2.findContours(gray, cv2.RETR_EXTERNAL,
                              cv2.CHAIN_APPROX_SIMPLE)  # discard/add the prefix _, before cnts, if it raise error
    count = 0
    for c in cnts:
        if cv2.contourArea(c) > 2500:  # and cv2.minEnclosingCircle(c)[1] >150:
            (x, y, w, h) = cv2.boundingRect(c)
            moments = cv2.moments(c)
            if moments['m00'] != 0:
                cx = int(moments['m10'] / moments['m00'])  # cx = M10/M00
                cy = int(moments['m01'] / moments['m00'])  # cy = M01/M00

            centr = (cx, cy)
            hight = 300
            width = 220

            tmp = frame[y:y+h, x:x + w]
            # cv2.imshow("temporary", tmp)

            framehight, framewidth, frmechannel = tmp.shape
            val =[]
            val=trimmer(tmp,framehight,framewidth )
            cv2.rectangle(frame, (x+val[0],y+val[1]),(x+val[0]+val[2], y+ val[1]+val[3])    ,(0, 120, 30),1)
            count = count + 1

    cv2.imshow("Full Window", frame)
    cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ =="__main__":
    entry()