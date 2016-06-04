from __future__ import division
import cv2
import math

# on the trimmer def, check the image iteratively for its pick till the image end and object end is below THRESHOLD
ref = cv2.imread("E1.jpg")
frame =cv2.imread("E2.jpg")
diff1 = cv2.absdiff(ref, frame) # be frame and ref free


thresh = cv2.threshold(diff1, 25, 255, cv2.THRESH_BINARY)[1]
# thresh = cv2.erode(thresh,None, iterations=3)
thresh = cv2.dilate(thresh,None, iterations=5)
gray= cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)


cv2.imshow("Ddiffata", diff1)
cv2.imshow("Data", gray)


(cnts, _) = cv2.findContours(gray, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)# discard/add the prefix _, before cnts, if it raise error
count =0
for c in cnts:
    if cv2.contourArea(c) > 1000: #and cv2.minEnclosingCircle(c)[1] >150:
        (x, y, w, h) = cv2.boundingRect(c)
        moments = cv2.moments(c)
        if moments['m00']!=0:
                    cx = int(moments['m10']/moments['m00']) # cx = M10/M00
                    cy = int(moments['m01']/moments['m00']) # cy = M01/M00
        centr =(cx,cy)

        hight =300
        width =200
        tmp = frame[y-10:y+int(w*1.15), x:x+w-10]
        framehight, framewidth,frmechannel =tmp.shape

        # cv2.circle(ref,centr,5,[255,255,255],2)
        cv2.rectangle(ref, (x-20, y), (x+w+20, int(y+((w)*1.25))), (0, 120, 30), 1) #change cx +constant based on area of object

        anotherimg= cv2.resize(tmp, None, fx=(width/framewidth),fy=(hight/framehight), interpolation=cv2.INTER_LINEAR)
        cv2.imshow("Resized%s"%count, anotherimg)
        count = count +1

cv2.imshow("Main", ref)
cv2.waitKey(0)

cv2.destroyAllWindows()