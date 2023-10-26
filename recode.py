import cv2
import numpy as np
import time
import os
import Cards
import VideoStream


IM_WIDTH = 1280
IM_HEIGHT = 720 
FRAME_RATE = 10

frame_rate_calc = 1
freq = cv2.getTickFrequency()

font = cv2.FONT_HERSHEY_SIMPLEX

# videostream = VideoStream.VideoStream((IM_WIDTH,IM_HEIGHT),FRAME_RATE,1,0).start()
time.sleep(1) # Give the camera time to warm up

path = os.path.dirname(os.path.abspath(__file__))
train_ranks = Cards.load_ranks( path + '/Card_Imgs/')
train_suits = Cards.load_suits( path + '/Card_Imgs/')

cam_quit = 0 
# =========================================================
# Grab frame from video stream
image = cv2.imread("T99.png")

# Pre-process camera image (gray, blur, and threshold it)
pre_proc = Cards.preprocess_image(image)
cv2.imshow("mask", pre_proc)
cv2.waitKey(0)
# Find and sort the contours of all cards in the image (query cards)
cnts_sort, cnt_is_card = Cards.find_cards(pre_proc)
# If there are no contours, do nothing
if len(cnts_sort) != 0:
    # Initialize a new "cards" list to assign the card objects.
    # k indexes the newly made array of cards.
    cards = []
    k = 0

    # For each contour detected:
    for i in range(len(cnts_sort)):
        if (cnt_is_card[i] == 1):

            cards.append(Cards.preprocess_card(cnts_sort[i],image))
            cards[k].best_rank_match,cards[k].best_suit_match,cards[k].rank_diff,cards[k].suit_diff = Cards.match_card(cards[k],train_ranks,train_suits)

            # Draw center point and match result on the image.
            image = Cards.draw_results(image, cards[k])
            k = k + 1
    # Draw card contours on image (have to do contours all at once or
    # they do not show up properly for some reason)
    if (len(cards) != 0):
        temp_cnts = []
        for i in range(len(cards)):
            temp_cnts.append(cards[i].contour)
        cv2.drawContours(image,temp_cnts, -1, (255,0,0), 2)
    
    

# Finally, display the image with the identified cards!
cv2.imshow("Card Detector",image)


# Poll the keyboard. If 'q' is pressed, exit the main loop.
cv2.waitKey(0)