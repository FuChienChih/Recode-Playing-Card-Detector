import cv2
import numpy as np
import os
import Cards


path = os.path.dirname(os.path.abspath(__file__))
train_ranks = Cards.load_ranks( path + '/Card_Imgs/')
train_suits = Cards.load_suits( path + '/Card_Imgs/')

cam_quit = 0 
# =========================================================
# Grab frame from video stream
image = cv2.imread("T99.png")

# 影像預處理
pre_proc = Cards.preprocess_image(image)

# 偵測卡牌
cnts_sort, cnt_is_card = Cards.find_cards(pre_proc)

# 沒偵測輪廓，不執行
if len(cnts_sort) != 0:
    cards = []
    k = 0

    # 從所有輪廓中找到牌，並將影像存入 cards
    for i in range(len(cnts_sort)):
        if (cnt_is_card[i] == 1):
            # 判斷card的size決定要不要再分割10/27   應該要對cnts_sort[i]做for處理，直到cnts_sort[i]的size小於一定的值
            x,y,w,h = cv2.boundingRect(cnts_sort[i])
            print('=============')
            temp_x,temp_y,temp_w,temp_h = x,y,w,h
            while temp_w*temp_h > 30*30:
                print('x,y,w,h',x,y,w,h)
                cv2.imshow("ori", image[y:y+h,x:x+w])
                print(f'ori({y}:{y+h},{x}:{x+w})')
                cv2.imshow("cut", image[temp_y:temp_y+13,x:x+w])
                print(f'cut({y}:{y+13.5},{x}:{x+w})')
                cv2.waitKey(0)
                temp_y += 13
                temp_h -= 13
                
            # 或許可以在preprocess_card中做遞迴，並將return設為list不用append用+
            cards.append(Cards.preprocess_card(cnts_sort[i],image))
            print(cards[k].contour)

            cards[k].best_rank_match,cards[k].best_suit_match,cards[k].rank_diff,cards[k].suit_diff = Cards.match_card(cards[k],train_ranks,train_suits)

            print('w,h = ',cards[k].width,cards[k].height)
            k = k + 1

    # 畫框
    if (len(cards) != 0):
        temp_cnts = []
        for i in range(len(cards)):
            temp_cnts.append(cards[i].contour)
        cv2.drawContours(image,temp_cnts, -1, (255,0,0), 2)
    
    

# 原圖展示
cv2.imshow("Card Detector",image)
cv2.waitKey(0)