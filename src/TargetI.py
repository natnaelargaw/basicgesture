import numpy as np
import math

import cv2

#Next Plan
'''
use wessis recognizer to interact with the neural network classifier
* those images with two or more contour point will be used to compose the training and test data
plus use it for real time image passing for the class
'''


def trimImage(data1):

    inputimg = data1.copy()
    w,h,c =inputimg.shape
    flag = 0
    morphD = 25 # iteration for dilating the image w/ the kernel
    extX = extY =extW = extH = 0 #This are contour points required to process incoming cropped frame.

    # Here image is top-right shifted by 5. This helps to detect significant edges quickly
    left = inputimg[0:w - 5, 0:h]  # vertically
    right = inputimg[0:5, 0:h]
    veritcal = np.concatenate((right, left), axis=0)  # vertically merge

    top = veritcal[0:w, 0:h - 5]
    bottom = veritcal[0:w, 0:5]
    final = np.concatenate((bottom, top), axis=1)  # vertically merge

    # The difference between shifted image and the original, Low Pass Filtering, and BG subtraction
    diff = cv2.absdiff(inputimg, final)
    blurred = cv2.GaussianBlur(diff, (35,35), 0)
    thresh = cv2.threshold(blurred, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=morphD)

    gray = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
    (cnts, _) = cv2.findContours(gray, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)


    for contour in cnts:
        (littleX, littleY, littleWidth, littleHight) = cv2.boundingRect(contour)
        if cv2.contourArea(contour) > 1000:

            if not flag:
                extX = littleX; extY = littleY; extW = littleWidth; extH = littleHight
                flag = 1
            else:
                if littleX <= 3 or littleY <= 3:  # Jumping extreme edges created by part-of-image shifting: usually at x and y beginning
                    continue
                elif extW < littleWidth and extH < littleHight:
                    extX = littleX;
                    extY = littleY;
                    extW = littleWidth;
                    extH = littleHight
        extH = int(extW * 1.15) # optional; but considers the portrait view of the hand
        if extH > 250: extH = int(extH / 2) #substitute this with extreme point detector function

        # to get rid of overshoots caused by dilation
        extW = extW - 10; extX = extX -5; extY = extY + 5

    return [extX, extY, extW, extH]


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


# An entry point that initiates first pass ROI identification and sends identified ROI frames to the second pass ROI finder/Trimmer.
def entryPoint():

    boundingpoints=[]
    # camera = cv2.VideoCapture("/home/ati/sam.webm")
    camera = cv2.VideoCapture(0)

    noOframes = 1
    ref = ''
    init = 0
    morphD = 15
    trimmedcontour = []

    while (camera.isOpened()):

        ret, frame = camera.read()
        # isolate the original frame
        draw = frame.copy()
        final = frame.copy()

        # take the very first frame as reference
        if not init:
            ref = frame
            init = 1
            cv2.imshow("Second Pass", final)
            continue

        if not noOframes % 50: # change this w/ 2* FPS
            ref = frame
            cv2.imshow("Second Pass", final)


        diff1 = cv2.absdiff(ref, frame)

        thresh = cv2.threshold(diff1, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=morphD)
        gray = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)

        (cnts, _) = cv2.findContours(gray, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        for c in cnts:
            if cv2.contourArea(c) > 2000:  # and cv2.minEnclosingCircle(c)[1] >150:
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(draw, (x, y), (x + w, y + h), (0, 120, 30), 1)
                tmp = frame[y:y + h, x:x + w]

                trimmedcontour = trimImage(tmp)
                boundingpoints.append((x + trimmedcontour[0],y + trimmedcontour[1],x + trimmedcontour[0] + trimmedcontour[2] +10, y + trimmedcontour[1] + trimmedcontour[3]))# recovery of eroded data


        # This is where second pass results are rendered to the current frame
        for ii in range(len(boundingpoints)):
            isolatedImg =frame[y + trimmedcontour[1]:y + trimmedcontour[1] + trimmedcontour[3],x + trimmedcontour[0]:x + trimmedcontour[0] + trimmedcontour[2]]

            wessiRecognizer(frame,isolatedImg)

            cv2.rectangle(final, (boundingpoints[ii][0],boundingpoints[ii][1]), (boundingpoints[ii][2],boundingpoints[ii][3]), (0, 127, 0), 2)

        boundingpoints = []

        # Display
        noOframes = noOframes + 1
        cv2.imshow("Second Pass", final)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    entryPoint()