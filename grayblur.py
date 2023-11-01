### Takes a card picture and creates a top-down 200x300 flattened image
### of it. Isolates the suit and rank and saves the isolated images.
### Runs through A - K ranks and then the 4 suits.

# Import necessary packages
import cv2
import numpy as np
import time
import Cards
import os

img_path = os.path.dirname(os.path.abspath(__file__)) + '/Card_Imgs/'

IM_WIDTH = 1280
IM_HEIGHT = 720

RANK_WIDTH = 70
RANK_HEIGHT = 125

SUIT_WIDTH = 70
SUIT_HEIGHT = 100

# If using a USB Camera instead of a PiCamera, change PiOrUSB to 2
PiOrUSB = 1



# Use counter variable to switch from isolating Rank to isolating Suit
i = 1

for Name in ['Ace','Two','Three','Four','Five','Six','Seven','Eight',
             'Nine','Ten','Jack','Queen','King','R_Ace','R_Two','R_Three',
             'R_Four','R_Five','R_Six','R_Seven','R_Eight','R_Nine','R_Ten','R_Jack',
             'R_Queen','R_King','Spades','Diamonds','Clubs','Hearts']:

    filename = Name + '.jpg'

    # print('Press "p" to take a picture of ' + filename)
    
    
    image = cv2.imread(f"Card_Imgs/{Name}.jpg")
    # Pre-process image
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # =============================================
    # 先將紅色轉為黑色(w: 造成影響辨識效果差)
    # bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # lower = np.array([0, 164, 0])
    # upper = np.array([255, 255, 255])
    # gray = cv2.inRange(bgr_image, lower, upper)
    # =============================================

    card = gray

    corner_zoom = cv2.resize(card, (0,0), fx=4, fy=4)
    corner_blur = cv2.GaussianBlur(corner_zoom,(5,5),0)

    # Isolate suit or rank
    if i <= 26: # Isolate rank
        rank_sized = cv2.resize(corner_blur, (RANK_WIDTH, RANK_HEIGHT), 0, 0)
        final_img = rank_sized

    if i > 26: # Isolate suit
        suit_sized = cv2.resize(corner_blur, (SUIT_WIDTH, SUIT_HEIGHT), 0, 0)
        final_img = suit_sized

    cv2.imshow("Image",final_img)
    retval,thresh = cv2.threshold(final_img,130,225,cv2.THRESH_BINARY)
    # Save image
    print('Press "c" to continue.')
    key = cv2.waitKey(0) & 0xFF
    if key == ord('c'):
        # name = img_path+filename
        cv2.imwrite(img_path+filename,final_img)

    i = i + 1

cv2.destroyAllWindows()