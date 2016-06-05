from collections import deque
import cv2
import numpy as np



def resizeforIP():
    return 0

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
    thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
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
            passh = passw
        if passh > 250: passh = int(passh / 2)  # to make unwanted hand-part discarded
        passw = passw - 5 * 2;
        passx = passx + 5;
        passy = passy + 5  # to get rid of dilation effects
        w1, h1, c1= data1.shape
    return [passx, passy, passw, passh]




def entry():
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

        diff1 = cv2.absdiff(ref, frame)  # be frame and ref free
        morphD = 15
        thresh = cv2.threshold(diff1, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=1)
        thresh = cv2.dilate(thresh, None, iterations=morphD)
        gray = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
        (cnts, _) = cv2.findContours(gray, cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)

        hight =300 ;  width =200

        for c in cnts:
            if cv2.contourArea(c) > 2000:  # and cv2.minEnclosingCircle(c)[1] >150:
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(draw, (x, y), (x + w, y + h), (0, 120, 30), 1)
                tmp = frame[y:y + h, x:x + w]
                framehight, framewidth, frmechannel = tmp.shape

                val = []
                val = trimImage(tmp, framehight, framewidth)
                cv2.rectangle(final, (x + val[0], y + val[1]), (x + val[0] + val[2], y + val[1] + val[3]), (0, 0, 255),
                              2)
                #display
        cv2.imshow("Draw", draw)
        cv2.imshow("Final", final)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    entry()