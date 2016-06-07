import cv2
import numpy as np
import math

#Next Plan
'''
use wessis recognizer to interact with the neural network classifier
* those images with two or more contour point will be used to compose the training and test data
plus use it for real time image passing for the class
'''


def trimImage(data1, w, h):
    inputimg = data1.copy()
    left = inputimg[0:w - 5, 0:h]  # vertically
    right = inputimg[0:5, 0:h]
    veritcal = np.concatenate((right, left), axis=0)  # vertically merge

    top = veritcal[0:w, 0:h - 5]
    bottom = veritcal[0:w, 0:5]
    final = np.concatenate((bottom, top), axis=1)  # vertically merge

    diff = cv2.absdiff(inputimg, final)  # be frame and ref free
    morphD = 25 #this one has direct proportionality to trimmer performance
    blurred = cv2.GaussianBlur(diff, (35,35), 0)
    thresh = cv2.threshold(blurred, 25, 255, cv2.THRESH_BINARY)[1]
    # thresh = cv2.erode(thresh, None, iterations=1)
    thresh = cv2.dilate(thresh, None, iterations=morphD)

    gray = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
    (cnts, _) = cv2.findContours(gray, cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)
    flag = passx = passy = passw = passh = 0  # initialize contour coordinates

    for cx in cnts:
        (xx2, yy2, ww2, hh2) = cv2.boundingRect(cx)
        if cv2.contourArea(cx) > 1000:
            if not flag:
                passx = xx2;
                passy = yy2;
                passw = ww2;
                passh = hh2
                flag = 1
            else:
                if xx2 <= 3 or yy2 <= 3:  # smothing shift edges
                    continue
                elif passw < ww2 and passh < hh2:
                    passx = xx2;
                    passy = yy2;
                    passw = ww2;
            passh = int(passw*1.15)
        if passh > 250: passh = int(passh / 2)  # to make unwanted hand-part discarded
        passw = passw - 5;
        passx = passx + 5;
        passy = passy + 5  # to get rid of dilation effects
        w1, h1, c1= data1.shape
    return [passx, passy, passw, passh]

def nn():
    pass


#Checking Wessi's Recognizer with  BIT modification
def wessiRecognizer(fulldata,cropdata):
    try:
        grey = cv2.cvtColor(cropdata, cv2.COLOR_BGR2GRAY)
        value = (35, 35)
        blurred = cv2.GaussianBlur(grey, value, 0)
        _, thresh1 = cv2.threshold(blurred, 127, 255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)



        contours, hierarchy = cv2.findContours(thresh1.copy(),cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        max_area = -1


        for i in range(len(contours)):
            cnt=contours[i]
            area = cv2.contourArea(cnt)
            if(area > max_area):
                max_area=area
                ci=i
        cnt=contours[ci]

        #######################This line draws the center#####################
        moments = cv2.moments(cnt)
        if moments['m00']!=0:
                    cx = int(moments['m10']/moments['m00']) # cx = M10/M00
                    cy = int(moments['m01']/moments['m00']) # cy = M01/M00

        centr=(cx,cy)
        # cv2.circle(cropdata,centr,5,[0,0,255],2)
        ######################This line draws the center######################
        x,y,w,h = cv2.boundingRect(cnt)
        # cv2.rectangle(cropdata,(x,y),(x+w,y+h),(0,0,255),0)
        hull = cv2.convexHull(cnt)
        drawing = np.zeros(cropdata.shape,np.uint8)



        cv2.drawContours(drawing,[cnt],0,(0,255,0),0)
        cv2.drawContours(drawing,[hull],0,(0,0,255),0)
        hull = cv2.convexHull(cnt,returnPoints = False)
        defects = cv2.convexityDefects(cnt,hull)
        count_defects = 0

        for i in range(defects.shape[0]):
            s,e,f,d = defects[i,0]
            start = tuple(cnt[s][0])
            end = tuple(cnt[e][0])
            far = tuple(cnt[f][0])
            a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
            c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
            angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57
            if angle <= 90:
                count_defects += 1
                cv2.circle(cropdata,far,1,[0,0,255],-1)
            elif angle >= 180 and angle >= 90:
                cv2.putText(fulldata, str, (30,100), cv2.FONT_HERSHEY_DUPLEX, 1.6, 1.6)
            cv2.line(cropdata, start, end, [0, 0, 0], 2)
        str =''
        if count_defects == 1:
            str = "Two fingers up"
            cv2.rectangle(cropdata,(x,y),(x+w,y+h),(0,0,255),0)
        elif count_defects == 2:
            str = "Three fingers up"
            cv2.rectangle(cropdata,(x,y),(x+w,y+h),(0,0,255),0)
        elif count_defects == 3:
            str = "Four fingers up"
            cv2.rectangle(cropdata,(x,y),(x+w,y+h),(0,0,255),0)
        elif count_defects == 4:
            str ="Five Fingers"
            cv2.rectangle(cropdata,(x,y),(x+w,y+h),(0,0,255),0)
        else:
            pass
            # str =". . . "
        if not str=='':
            print(str)
        cv2.putText(fulldata, str, (30,100), cv2.FONT_HERSHEY_COMPLEX, 1.2, 1.2)
        cv2.imshow("Wessis Result",fulldata)
        # cv2.imshow("Wessis Cropped",cropdata)
        # cv2.imshow("Wessis Thresh",thresh1)#this affects the wessis out put

    except:
        pass

def entryPoint():
    boundingpoints=[]
    # camera = cv2.VideoCapture("/home/ati/sa1.mp4")
    camera = cv2.VideoCapture(0)
    count1 = 1
    ref = ''
    flag = 0

    while True:
        ret, frame = camera.read()
        draw = frame.copy()
        final = frame.copy()
        if not flag:
            ref = frame
            flag = 1
            continue

        if not count1 % 7:
            ref = frame

        count1 = count1 + 1
        diff1 = cv2.absdiff(ref, frame)
        morphD = 15
        thresh = cv2.threshold(diff1, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=morphD)
        gray = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)

        (cnts, _) = cv2.findContours(gray, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        hight =300 ;  width =200

        for c in cnts:
            if cv2.contourArea(c) > 2000:  # and cv2.minEnclosingCircle(c)[1] >150:
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(draw, (x, y), (x + w, y + h), (0, 120, 30), 1)
                tmp = frame[y:y + h, x:x + w]
                framehight, framewidth, frmechannel = tmp.shape

                val = []
                val = trimImage(tmp, framehight, framewidth)
                boundingpoints.append((x + val[0],y + val[1],x + val[0] + val[2] +10, y + val[1] + val[3]))# recovery of eroded data
                sendme =frame[y + val[1]:y + val[1] + val[3],x + val[0]:x + val[0] + val[2]]
                # try:
                #     cv2.imshow("Wessi",sendme)
                # except:
                #     pass
                wessiRecognizer(frame,sendme)

        for ii in range(len(boundingpoints)):
            cv2.rectangle(final, (boundingpoints[ii][0],boundingpoints[ii][1]), (boundingpoints[ii][2],boundingpoints[ii][3]), (0, 127, 0), 2)

        boundingpoints = []

        #display
        # cv2.imshow("Draw", draw)
        cv2.imshow("Final", final)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    entryPoint()