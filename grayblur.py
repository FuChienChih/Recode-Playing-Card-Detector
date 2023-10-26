import cv2
import numpy as np

BKG_THRESH = 60
image = cv2.imread("T99.png")


gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray,(1,1),0)


img_w, img_h = np.shape(image)[:2]
bkg_level = gray[int(img_h/100)][int(img_w/2)]
thresh_level = bkg_level + BKG_THRESH

retval, thresh = cv2.threshold(blur,thresh_level,255,cv2.THRESH_BINARY)

print(gray)
cv2.imshow("gray", gray)
cv2.imshow("blur", blur)
cv2.waitKey(0)