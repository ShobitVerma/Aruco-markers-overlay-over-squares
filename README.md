import cv2
from cv2 import approxPolyDP
import cv2.aruco as aruco
import numpy as np
import arucodetec as ad
import math
import cvzone


def rescaleFrame(frame,scale): # for resisezing image
    width = int(frame.shape[1]*scale)
    height = int(frame.shape[0]*scale)

    dimensions = (width,height)
    return cv2.resize(frame,dimensions,interpolation= cv2.INTER_AREA)


img=cv2.imread('CVtask.jpg')
aru1 = cv2.imread("Ha.jpg")
aru2 = cv2.imread("HaHa.jpg")
aru3 = cv2.imread("LMAO.jpg")
aru4 = cv2.imread("XD.jpg")


aru= [aru1,aru2,aru3,aru4]


blank = np.zeros((img.shape), dtype='uint8')
#cv2.imshow("blank",blank)

gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
res, thres = cv2.threshold(gray,230,255,cv2.THRESH_BINARY)#converting into binary img


contour,xtra=cv2.findContours(thres,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)#matrix with point values
square =0
for cont in contour:
    approx=cv2.approxPolyDP(cont,0.01*cv2.arcLength(cont,True),True)#approximate poly



    if (len(approx)==4):#quardilateral detected
        x1 = approx.ravel()[0]
        y1 = approx.ravel()[1]

        x2 = approx.ravel()[2]
        y2 = approx.ravel()[3]

        x3 = approx.ravel()[4]
        y3 = approx.ravel()[5]

        x4 = approx.ravel()[6]
        y4 = approx.ravel()[7]

        centerx =int((x1+x3)/2)
        centery =int ((y1+y3)/2)


        midy=(y2+y3)/2
        midx=(x2+x3)/2

         
        xp1,yp1,wid,hei=cv2.boundingRect(approx)



        if((float(wid)/hei)<1.05 and (float(wid)/hei)>0.95): #for square
            slope = (midy - centery) / (midx - centerx)
            angle = ((math.atan(slope)) * 180) / (np.pi)

            sidelength = int(math.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)))

            #cv2.drawContours(img,[approx],0,(0,200,0),2)


            #cv2.circle(img,(centerx,centery),0,(0,0,0),4)
            if (img[centery, centerx][0]) == 79 and (img[centery, centerx][1]) == 209 and (img[centery, centerx][2]) == 146:
                square=1

                #cv2.putText(img,"sq 1",(centerx,centery),cv2.FONT_HERSHEY_COMPLEX,3,(109,65,187))
            if (img[centery, centerx][0]) == 9 and (img[centery, centerx][1]) == 127 and (img[centery, centerx][2]) == 240:
                square = 2
                angle=0
                #cv2.putText(img,"sq 2",(centerx,centery),cv2.FONT_HERSHEY_COMPLEX,3,(109,65,187))

            if (img[centery, centerx][0]) == 0 and (img[centery, centerx][1]) == 0 and (img[centery, centerx][2]) == 0:
                square = 3

                #cv2.putText(img,"sq 3",(centerx,centery),cv2.FONT_HERSHEY_COMPLEX,3,(109,65,187))

            if (img[centery, centerx][0]) == 210 and (img[centery, centerx][1]) == 222 and (img[centery, centerx][2]) == 228:
                square = 4

                #cv2.putText(img,"sq 4",(centerx,centery),cv2.FONT_HERSHEY_COMPLEX,3,(109,65,187))
            cv2.drawContours(img, [approx], 0, (0,0,0), -1)
            for i in range(0,4):
                (id,corner)= ad.arucoinfo(aru[i])
                idp =id.flatten()
                cornerp=corner.reshape((4,2))

                a,b,c=img.shape

                if(idp==square):
                    acrop=ad.arucocrop(aru[i],sidelength)
                    (arot,centerax,centeray)=ad.arucoRotate(acrop,-angle)
                    afinal=ad.arucocrop2(arot,wid,centerax)
                    #afinalimg = rescaleFrame(afinal, 0.6)
                    blank[centery - int(afinal.shape[0] / 2):centery + int(afinal.shape[0] / 2),centerx - int(afinal.shape[0] / 2):centerx + int(afinal.shape[0] / 2)] = afinal



           
img = cv2.add(img,blank)
finalimg = rescaleFrame(img,0.6)

cv2.imshow("final",finalimg)



cv2.waitKey(0)
