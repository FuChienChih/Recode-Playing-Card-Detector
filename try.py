from PIL import ImageGrab
import numpy as np
import cv2
import Cards

def find_hand_rank_suit(denoise_image,org_image):
    # 找出所有輪廓值，階級（輪廓內還有多少輪廓），儲存在index_sort
    _, denoise_image = cv2.threshold(denoise_image, 100, 255, cv2.THRESH_BINARY_INV)  # 使用反转二值化
    cnts,hier = cv2.findContours(denoise_image,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    # If there are no contours, do nothing
    if len(cnts) == 0:
        return [], []

    rank_suit = []
    for i in range(len(cnts)):
        x,y,w,h = cv2.boundingRect(cnts[i])
        if  20 > w > 5 and 20 > h > 5:
            rank_suit.append(org_image[y:y+h,x:x+w])
            cv2.imshow('..',org_image[y:y+h,x:x+w])
            cv2.waitKey(0)
    return rank_suit[-1] ,rank_suit[-2]

img_rgb = cv2.imread("3213123.png")
img_bgr = cv2.cvtColor(np.array(img_rgb), cv2.COLOR_RGB2BGR)

org_image = img_bgr
org_image = cv2.resize(org_image, (1980, 988), interpolation=cv2.INTER_CUBIC)
denoise_image = Cards.denoise_image(img_bgr)
denoise_image = cv2.resize(denoise_image, (1980, 988), interpolation=cv2.INTER_CUBIC)
    

hands = Cards.find_cards(denoise_image) # 找出所有手牌的座標
x,y,w,h = hands[0]

hand = denoise_image[y:y+h,x:x+w]
hand2 = org_image[y:y+h,x:x+w]

# Cards.preprocess_card(x,y,w,h,denoise_image,org_image)
rank,suit = find_hand_rank_suit(hand,hand2)
cv2.imshow('rank',rank)
cv2.imshow('suit',suit)
cv2.waitKey(0)