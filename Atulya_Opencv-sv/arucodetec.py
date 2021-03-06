import cv2
import numpy as np
import cv2.aruco as aruco
import math
def rescaleFrame(frame,scale): # for resisezing image
    width = int(frame.shape[1]*scale)
    height = int(frame.shape[0]*scale)

    dimensions = (width,height)
    return cv2.resize(frame,dimensions,interpolation= cv2.INTER_AREA)

a=cv2.imread('Ha.jpg')

def arucoinfo(img):
    arucoDict = aruco.Dictionary_get(aruco.DICT_5X5_250)
    para = aruco.DetectorParameters_create()
    (corners, ids, r) = aruco.detectMarkers(img, arucoDict, parameters = para)
    return(ids, np.int0(corners))

def arucocrop(img,sidelength):
    id,corner= arucoinfo(img)
    idp=id.flatten()
    cr=corner.reshape(4,2)
    #cv2.drawContours(img, corner, 0, (255, 255, 255), 5)
    pt1 = np.float32(cr)
    pt2 = np.float32([[0, 0], [sidelength, 0], [sidelength, sidelength], [0, sidelength]])
    matrix= cv2.getPerspectiveTransform(pt1,pt2)
    output=cv2.warpPerspective(img,matrix,(sidelength,sidelength))

    
    return output

def arucocrop2(img,wid,centerax):#cropping with bounding box length
    sidelength=wid
    shift=centerax-wid/2
    pt1= np.float32([[shift, shift], [shift+sidelength, shift], [shift+sidelength,shift+sidelength], [shift,shift+sidelength]])
    pt2 = np.float32([[0, 0], [sidelength, 0], [sidelength, sidelength], [0, sidelength]])
    matrix= cv2.getPerspectiveTransform(pt1,pt2)
    output=cv2.warpPerspective(img,matrix,(sidelength,sidelength))

    
    return output


def arucoRotate(img,angle):
    img=rescaleFrame(img,2)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    
    rows,coloumns,channels=img.shape
    rotmatrix= cv2.getRotationMatrix2D((coloumns/2,rows/2),angle,0.5)
    output=cv2.warpAffine(img,rotmatrix,(coloumns,rows))
    output=cv2.cvtColor(output,cv2.COLOR_RGB2BGR)
    return (output,coloumns/2,rows/2)

