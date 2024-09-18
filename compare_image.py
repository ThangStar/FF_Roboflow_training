import numpy as np 
import cv2 
 
img = cv2.imread('p4.jpg') 
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
 
ret,thresh = cv2.threshold(gray,127,255,1) 
 
contours,h = cv2.findContours(thresh,1,2) 
 
for cnt in contours: 
    approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True) 
    if len(approx)==5: 
        cv2.drawContours(img,[cnt],0,255,-1) 
    elif len(approx)==3: 
        cv2.drawContours(img,[cnt],0,(0,255,0),-1) 
    elif len(approx)==4: 
        cv2.drawContours(img,[cnt],0,(0,0,255),-1) 
    elif len(approx) == 9: 
        cv2.drawContours(img,[cnt],0,(255,255,0),-1) 
    elif len(approx) > 15: 
        cv2.drawContours(img,[cnt],0,(0,255,255),-1) 
 
cv2.imshow('img',img) 
cv2.waitKey(0) 
cv2.destroyAllWindows() 