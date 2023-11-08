from PIL import ImageGrab
import numpy as np
import cv2
import Cards

def find_hand_rank_suit(denoise_image,org_image):
    # 找出所有輪廓值，階級（輪廓內還有多少輪廓），儲存在index_sort
    _, denoise_image = cv2.threshold(denoise_image, 100, 255, cv2.THRESH_BINARY_INV)  # 使用反转二值化
    
    cnts,hier = cv2.findContours(denoise_image,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

    x,y,w,h = cv2.boundingRect(cnts[0])

    for i in range(len(cnts)):
        x,y,w,h = cv2.boundingRect(cnts[i])
        print(f'w:{w},h:{h}')
        # 找到花色
        if  20 > w > 5 and 20 > h > 5:
            cv2.imshow('finded_card',org_image[y:y+h,x:x+w])
            cv2.waitKey(0)
        # 找到數字
        if True:
            pass
    cv2.imshow('denoise_image',denoise_image)
    cv2.waitKey(0)
    # If there are no contours, do nothing
    if len(cnts) == 0:
        return 

img_rgb = cv2.imread("img_nor2.png")
img_bgr = cv2.cvtColor(np.array(img_rgb), cv2.COLOR_RGB2BGR)

org_image = img_bgr

denoise_image = Cards.denoise_image(img_bgr)

# denoise_image = cv2.resize(denoise_image, (1980, 988), interpolation=cv2.INTER_CUBIC)

hands = Cards.find_cards(denoise_image) # 找出所有手牌的座標
x,y,w,h = hands[4]

org_hand = org_image[y:y+h,x:x+w]
denoise_hand = denoise_image[y:y+h,x:x+w]

find_hand_rank_suit(denoise_hand,org_hand)


# cv2.imshow('denoise_image',denoise_image)
# cv2.imshow('hands',org_image[y:y+h,x:x+w])
# cv2.waitKey(0)
