import cv2
import numpy as np
import os
import Cards

RANK_WIDTH = 38

path = os.path.dirname(os.path.abspath(__file__))
train_ranks = Cards.load_ranks( path + '/Card_Imgs/')
train_suits = Cards.load_suits( path + '/Card_Imgs/')

cam_quit = 0 
# =========================================================
# Grab frame from video stream
image = cv2.imread("TT0.png")
image2 = cv2.imread("img.png")
# 影像預處理
pre_proc = Cards.preprocess_image(image)
# pre_proc = Cards.preprocess_image(image2)

# 偵測卡牌
cnts_sort, cnt_is_card = Cards.find_cards(pre_proc)

# 沒偵測輪廓，不執行
if len(cnts_sort) != 0:
    list_cards = []
    k = 0

    # 從所有輪廓中找到牌，並將影像存入 cards
    for i in range(len(cnts_sort)):
        if (cnt_is_card[i] == 1):
            # 判斷card的size決定要不要再分割10/27
            x,y,w,h = cv2.boundingRect(cnts_sort[i])
            # 將牌至中
            w-=2
            x+=1
            y+=1
            # print(x,y,w,h)
            temp_x,temp_y,temp_w,temp_h = x,y,w,h
            # 將牌分割
            # while temp_w*temp_h > 31*23 and w < 35:
            #     cards.append(Cards.preprocess_card(x,temp_y,w,13,image2))
            #     print(x,temp_y,w,14)
            #     temp_y += 13
            #     temp_h -= 13
            if  w < 35:
                 list_cards+=Cards.preprocess_card(x,temp_y,w,h,image2)


            k = k + 1
    for card in list_cards:
        card.best_rank_match,card.best_suit_match,card.rank_diff,card.suit_diff = \
        Cards.match_card(card,train_ranks,train_suits)
        print('best_rank_match',card.best_rank_match,'\nbest_suit_match',card.best_suit_match)
    
    # print('cards[0]',cards[0])
    # 畫框
    # if (len(cards) != 0):
    #     temp_cnts = []
    #     for i in range(len(cards)):
    #         temp_cnts.append(cards[i].contour)
    #     cv2.drawContours(image,temp_cnts, -1, (255,0,0), 2)
    
    

# 原圖展示
# cv2.imshow("Card Detector",image)
cv2.waitKey(0)