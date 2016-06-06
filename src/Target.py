from __future__ import division
import cv2
import numpy as np
import math
#--------------------------------------------------------------
# Tested this approach w/ even cropped image. it sucks
def WessisGesturRecognition(croppedData):
    grey = cv2.cvtColor(croppedData, cv2.COLOR_BGR2GRAY)
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
    x,y,w,h = cv2.boundingRect(cnt)
    cv2.rectangle(croppedData,(x,y),(x+w,y+h),(0,0,255),0)
    hull = cv2.convexHull(cnt)
    drawing = np.zeros(croppedData.shape,np.uint8)
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
        if angle <= 45:
            count_defects += 1
            cv2.circle(croppedData,far,1,[0,0,255],-1)
        elif angle >= 180 and angle >= 90:
            cv2.putText(croppedData, str, (30,100), cv2.FONT_HERSHEY_DUPLEX, 1.6, 1.6)
        cv2.line(croppedData, start, end, [0, 0, 0], 2)
        if count_defects == 1:
            str = "Two fingers up"
        elif count_defects == 2:
            str = "Three fingers up"
        elif count_defects == 3:
            str = "Four fingers up"
        elif count_defects == 4:
            str= "\"Hi hi...\" "
        else:
            str = "Recognizing Hand Gesture..."
        print(str)

#--------------------------------------------------------------




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
    inputimg= inputimg[passy:passy+passh, passx:passx+passw]

    #-----------------------------------------------------------------------------------------------------------------
    w,h,c=inputimg.shape
    left = inputimg[0:w-2, 0:h] #vertically
    right= inputimg[0:2, 0:h]
    veritcal = np.concatenate((right, left), axis=0) #vertically merge

    top = veritcal[0:w, 0:h-2]
    bottom = veritcal[0:w, 0:2]
    final= np.concatenate((bottom, top), axis=1) #vertically merge

    tmpdiff = cv2.absdiff(inputimg, final)  # be frame and ref free
    tmpthresh = cv2.dilate(tmpdiff, None, iterations=2)
    tmpthresh = cv2.threshold(tmpthresh, 20, 255, cv2.THRESH_BINARY)[1]
    tmpgray = cv2.cvtColor(tmpthresh, cv2.COLOR_BGR2GRAY) #better to take this as feature vector

    (conts, _) = cv2.findContours(tmpgray, cv2.RETR_EXTERNAL,
                           cv2.CHAIN_APPROX_SIMPLE)  # discard/add the prefix _, before cnts, if it raise error
    flag2=passx2 = passy2 = passw2 =passh2 = 0
    for cx in conts:
        (xx2, yy2, ww2, hh2) = cv2.boundingRect(cx)
        if cv2.contourArea(cx) > 1000:
            if not flag2:
                passx2=xx2;passy2=yy2;passw2=ww2; passh2=hh2
                flag2=1

            if passw2 < ww2 and passh2 < hh2:
                passx2=xx2;passy2=yy2;passw2=ww2;passh2=hh2
        passw = passw -(passw-passw2)  ;

    passx= passx +passx2 ; passy =passy +passy2; passh =passh -(passh-passh2) # to get rid of dilation effects
    cv2.rectangle(inputimg, (passx2,passy2),(passx2+passw2, passy2+passh2)    ,(0, 120, 30),1)
    #-----------------------------------------------------------------------------------------------------------------

    return [passx,passy, passw, passh]



def entry(ref, frame):
    # ref = cv2.imread("/home/ati/basicgesture/corpus/A1.jpg")
    # frame = cv2.imread(q"/home/ati/basicgesture/corpus/A2.jpg")
    diff1 = cv2.absdiff(ref, frame)  # be frame and ref free
    morphD=15
    thresh = cv2.threshold(diff1, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=1)
    thresh = cv2.dilate(thresh, None, iterations=morphD)
    gray = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
    (cnts, _) = cv2.findContours(gray, cv2.RETR_EXTERNAL,
                              cv2.CHAIN_APPROX_SIMPLE)  # discard/add the prefix _, before cnts, if it raise error
    for c in cnts:
        if cv2.contourArea(c) > 2500:  # and cv2.minEnclosingCircle(c)[1] >150:
            (x, y, w, h) = cv2.boundingRect(c)
            tmp = frame[y:y+h, x:x + w]
            framehight, framewidth, frmechannel = tmp.shape

            val =[]
            val=trimmer(tmp,framehight,framewidth )

            cv2.rectangle(frame, (x+val[0],y+val[1]),(x+val[0]+val[2], y+ val[1]+val[3])    ,(0, 120, 30),1)

            # Take Image from Here
            if y+val[1]== y+ val[1]+val[3]: # no image to isolate
                continue
            InputData = frame[y+val[1]:y+ val[1]+val[3],x+val[0]:x+val[0]+val[2]]
            print(y+val[1], y+ val[1]+val[3],x+val[0],x+val[0]+val[2])
            cv2.imshow("Input-Data",InputData)

            WessisGesturRecognition(InputData)


    cv2.imshow("Full Window", frame)
    cv2.waitKey(0)
    # cv2.destroyAllWindows()

def init():
    camera =cv2.VideoCapture(0)
    count1=1
    reff =''
    flag=0
    while True:

        # Get the current frame
        ret, frame = camera.read()
        if not flag:
            reff =frame
            flag=1
            continue
        if not count1 %7:
            reff=frame
        entry(reff, frame)
        count1 =count1 +1


if __name__ =="__main__":
    init()

