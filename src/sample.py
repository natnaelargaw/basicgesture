import cv2
import numpy as np
import math

#Next Plan
'''
use wessis recognizer to interact with the neural network classifier
* those images with two or more contour point will be used to compose the training and test data
plus use it for real time image passing for the class
'''


def entryPoint():
    boundingpoints=[]
    # camera = cv2.VideoCapture("/home/ati/sam.webm")
    camera = cv2.VideoCapture(0)
    count1 = 1
    ref = ''
    flag = 0

    while True:
        try:

            ret, frame = camera.read()
            gray =cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            if not flag:
                ref=gray
                flag =1
                continue
            diff= cv2.absdiff(ref,gray)
            val =cv2.threshold(diff, 10, 255,cv2.THRESH_BINARY)[1]
            val = cv2.GaussianBlur(val, (7,7), 0)
            val=cv2.dilate(val,None,3)


            (conts, _) = cv2.findContours(val, cv2.RETR_EXTERNAL,
                             cv2.CHAIN_APPROX_SIMPLE)
            for cx in conts:
                (x, y, w, h) = cv2.boundingRect(cx)
                # res = ((w-x)/(h-y))
                # print(res)
                if cv2.contourArea(cx)>500 and (w-x >10 and h-y >10):
                    cv2.drawContours(frame,[cx],0,(0,255,0),0)
                    cv2.rectangle(frame, (x,y),(x+w, y+h),(0, 120, 30),1)

            # for i in range(len(diff)-300):
            #     for j in range(len(diff[i])-300):
            #         if diff[i][j] <10:
            #             diff[i][j] =0
            #         else:
            #             diff[i][j] =255


            ref =gray
            cv2.imshow("Thresh", val)
            cv2.imshow("Final", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
        except:
            # camera = cv2.VideoCapture("/home/ati/sam.webm")
            camera = cv2.VideoCapture(0)
            continue
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    entryPoint()