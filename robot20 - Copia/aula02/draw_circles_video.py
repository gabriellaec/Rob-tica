#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__      = "Matheus Dib, Fabio de Miranda"

import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
import math
import matplotlib.cm as cm

# If you want to open a video, just change v2.VideoCapture(0) from 0 to the filename, just like below
#cap = cv2.VideoCapture('hall_box_battery.mp4')

# Parameters to use when opening the webcam.
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

lower = 0
upper = 1

def comp(a,b):
    if a[1] > b[1]:
        return -1
    elif a[1] == b[1]:
        return 0
    else:
        return 1


print("Press q to QUIT")

# Returns an image containing the borders of the image
# sigma is how far from the median we are setting the thresholds
def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged


 
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # A gaussian blur to get rid of the noise in the image
    blur = cv2.GaussianBlur(gray,(5,5),0)
    #blur = gray
    # Detect the edges present in the image
    bordas = auto_canny(blur)


    circles = []

    out_circles = []


    # Obtains a version of the edges image where we can draw in color
    bordas_color = cv2.cvtColor(bordas, cv2.COLOR_GRAY2BGR)

     


#cria mascara magenta
    magenta_menor = np.array([166,  50,  50])
    magenta_maior = np.array([176, 255, 255])
    mask_coke_mag = cv2.inRange(hsv, magenta_menor, magenta_maior)

#cria mascara ciano
    ciano_menor = np.array([100, 50, 50])
    ciano_maior = np.array([110, 255, 255])
    mask_coke_cian = cv2.inRange(hsv, ciano_menor, ciano_maior)

    masks = mask_coke_mag + mask_coke_cian

    imagem = cv2.bitwise_or(frame, frame, mask=masks)
    imagem2 = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    #cv2.imshow("mascara", imagem)

      

    # HoughCircles - detects circles using the Hough Method. For an explanation of
    # param1 and param2 please see an explanation here http://www.pyimagesearch.com/2014/07/21/detecting-circles-images-using-opencv-hough-circles/
    circles = None
    circles=cv2.HoughCircles(imagem2,cv2.HOUGH_GRADIENT,2,70,param1=50,param2=60,minRadius=5,maxRadius=100)


  

    if circles is not None:        
        circles = np.uint16(np.around(circles)).astype("int")
        for i in circles[0,:]:
            print(i)
            # draw the outer circle
            # cv2.circle(img, center, radius, color[, thickness[, lineType[, shift]]])
           
           
        #if i[2]>=20 and i[2]<60:
           ############################################
            cv2.circle(bordas_color,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv2.circle(bordas_color,(i[0],i[1]),i[2],(0,0,255),3)

            out_circles.append( ((i[0], i[1]) , i[2]) )

            if len(out_circles)>=2:
                x, y = out_circles[0][0]
                x2, y2 = out_circles[1][0]
               # x = out_circles[0][0]
                #x2 = out_circles[1][0]
                #y = out_circles[0][1]
                #y2 = out_circles[1][1]

                print(x, x2, y, y2)
                font = cv2.FONT_HERSHEY_SIMPLEX

                #distancia entre os dois circulos
                deltaX=(x-x2)**2
                deltaY=(y-y2)**2
                h=(deltaX+deltaY)**(0.5)
               # cv2.putText(imagem, "h: {}".format(h) ,(200,200), font, (0.5),(255,255,255),2,cv2.LINE_AA)

                H=14
                f=707
                #distancia entre folha e camera
                D=(H*f)/h

                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(imagem, "Distancia: {}".format(D) ,(0,450), font, (0.75),(255,255,255),2,cv2.LINE_AA)

                #linha entre circulos

                cv2.line(imagem, (x,y), (x2,y2), (0, 255, 0), thickness=3, lineType=8)

                #angulo entre linha entre circulos e a horizontal
                Y=abs(y-y2)
                X=abs(x-x2)
                angulo=math.atan2(Y,X)
                angulo=math.degrees(angulo)

                #print(angulo)
                cv2.putText(bordas_color,"Angulo: {}".format(angulo),(0,400), font, (0.75),(255,255,255),2,cv2.LINE_AA)


            #cv2.imshow('img', img)


    out_circles.sort(cmp = comp)

    print(out_circles)



    # Draw a diagonal blue line with thickness of 5 px
    # cv2.line(img, pt1, pt2, color[, thickness[, lineType[, shift]]])
    #cv2.line(bordas_color,(0,0),(511,511),(255,0,0),5)

    # cv2.rectangle(img, pt1, pt2, color[, thickness[, lineType[, shift]]])
    #cv2.rectangle(bordas_color,(384,0),(510,128),(0,255,0),3)

    # cv2.putText(img, text, org, fontFace, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]])
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(bordas_color,'Press q to quit',(0,50), font, 1,(255,255,255),2,cv2.LINE_AA)

    #More drawing functions @ http://docs.opencv.org/2.4/modules/core/doc/drawing_functions.html

    # Display the resulting frame
    #cv2.imshow('Detector de circulos',bordas_color)
    #cv2.imshow('Frame',frame)
 
 
    tudo = bordas_color + imagem 

    cv2.imshow('Tudo', tudo)



    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#  When everything done, release the capture
cap.release()
cv2.destroyAllWindows()