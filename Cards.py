############## Playing Card Detector Functions ###############
#
# Author: Evan Juras
# Date: 9/5/17
# Description: Functions and classes for CardDetector.py that perform 
# various steps of the card detection algorithm


# Import necessary packages
import numpy as np
import cv2
import time
import os
path = os.path.dirname(os.path.abspath(__file__))


### Constants ###

# Adaptive threshold levels
BKG_THRESH = 60
CARD_THRESH = 30

# Width and height of card corner, where rank and suit are
CORNER_WIDTH = 84
CORNER_HEIGHT = 120

# Dimensions of rank train images
RANK_WIDTH = 70
RANK_HEIGHT = 125

# Dimensions of suit train images
SUIT_WIDTH = 70
SUIT_HEIGHT = 100

RANK_DIFF_MAX = 2000
SUIT_DIFF_MAX = 700

CARD_MAX_AREA = 4000
CARD_MIN_AREA = 90

font = cv2.FONT_HERSHEY_SIMPLEX

### Structures to hold query card and train card information ###

class Query_card:
    """
    這個類別用於儲存單張卡片的相關屬性。

    Attributes:
        - contour : 卡片的輪廓。
        - warp : 38x18大小的、灰階、模糊處理過的圖像。
        - rank_img : 卡片點數圖像。
        - suit_img : 卡片花色圖像。
        - best_rank_match : 最佳匹配點數。
        - best_suit_match : 最佳匹配花色。
        - rank_diff : 點數圖像與訓練圖像之間的差異。
        - suit_diff : 花色圖像與訓練圖像之間的差異。
    """

    def __init__(self):
        self.contour = []
        self.warp = []
        self.rank_img = []
        self.suit_img = []
        self.best_rank_match = "Unknown"
        self.best_suit_match = "Unknown"
        self.rank_diff = 0
        self.suit_diff = 0

class Train_ranks:
    """用於儲存訓練圖像(點數)"""

    def __init__(self):
        self.img = []
        self.name = "Placeholder"

class Train_suits:
    """用於儲存訓練圖像(花色)"""

    def __init__(self):
        self.img = []
        self.name = "Placeholder"

### Functions ###
def load_ranks(filepath):
    """
    從指定路徑載入訓練用圖像 後續用於判斷Qcard物件的"最佳匹配數字"。

    Parameters:
        - filepath (str): 點數圖像的檔案路徑。

    Returns:
        - train_ranks (list): 包含Train_ranks物件的list。
    """
    train_ranks = []
    i = 0
    
    for Rank in ['Ace','Two','Three','Four','Five','Six','Seven','Eight',
                'Nine','Ten','Jack','Queen','King','R_Ace','R_Two','R_Three',
                'R_Four','R_Five','R_Six','R_Seven','R_Eight','R_Nine','R_Ten','R_Jack',
                'R_Queen','R_King']:
        train_ranks.append(Train_ranks())
        train_ranks[i].name = Rank
        filename = Rank + '.jpg'
        train_ranks[i].img = cv2.imread(filepath+filename, cv2.IMREAD_GRAYSCALE)
        i = i + 1

    return train_ranks

def load_suits(filepath):
    """
    從指定路徑載入訓練用圖像 後續用於判斷Qcard物件的"最佳匹配花色"。

    Parameters:
        - filepath (str): 點數圖像的檔案路徑。

    Returns:
        - train_suits (list): 包含Train_suits物件的list。
    """

    train_suits = []
    i = 0
    
    for Suit in ['Spades','Diamonds','Clubs','Hearts']:
        train_suits.append(Train_suits())
        train_suits[i].name = Suit
        filename = Suit + '.jpg'
        train_suits[i].img = cv2.imread(filepath+filename, cv2.IMREAD_GRAYSCALE)
        i = i + 1

    return train_suits

def find_cards(thresh_image):
    """此函式根據輪廓size大到小排列 並判斷輪廓是否包含手牌輪廓的各種特徵"""

    # 找出所有輪廓值，階級（輪廓內還有多少輪廓），儲存在index_sort
    cnts,hier = cv2.findContours(thresh_image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # If there are no contours, do nothing
    if len(cnts) == 0:
        return [], []

    # 輪廓值，階級 由低到高儲存進 cnts_sort hier_sort
    index_sort = sorted(range(len(cnts)), key=lambda i : cv2.contourArea(cnts[i]),reverse=True)
    cnts_sort = []
    hier_sort = []
    cnt_is_card = np.zeros(len(cnts),dtype=int)
    for i in index_sort:
        cnts_sort.append(cnts[i])
        hier_sort.append(hier[0][i])

    for i in range(len(cnts_sort)):
        size = cv2.contourArea(cnts_sort[i])
        peri = cv2.arcLength(cnts_sort[i],True) # 是否閉合
        approx = cv2.approxPolyDP(cnts_sort[i],0.02*peri,True) # 方形容錯率0.02
        if ((size < CARD_MAX_AREA) and (size > CARD_MIN_AREA)
            and (hier_sort[i][3] == -1) and (len(approx) == 4)):
            
            cnt_is_card[i] = 1

    return cnts_sort, cnt_is_card

def preprocess_card(x,y,w,h,image):
    """
    預處理卡片圖像。

    Parameters:
    - x (int): 手牌的左上角 x 坐标。
    - y (int): 手牌的左上角 y 坐标。
    - w (int): 手牌的宽度。
    - h (int): 手牌的高度。
    - image (numpy.ndarray): 输入的图像，通常是 BGR 格式的图像。

    Returns:
    - qCard_list : 存放卡片物件的list
    - 卡片物件 : 包含每一張單獨卡片的座標,中心,數字圖像,花色圖像...等屬性

    Description:
    此方法用來預處理"手牌"(一張或多張撲克牌)
    將手牌分割成多張“卡片”(card)
    預處理包括裁剪、大小調整 以確保卡片圖像有一致的格式和特徵
    """
    qCard_list = []
    # 將手牌調整成預設大小
    cards = image[y:y+h,x:x+w]
    cards_w = 38
    cards_h = int(h*38/w)
    cards = cv2.resize(cards, (cards_w, cards_h),0,0)
    
    # 要分割的卡片x,y座標
    x_card,y_card = x,y

    # 以手牌座標為基準,卡片的y座標
    last_y = 0
    while cards_h > 30 :
        # 建立一個卡片物件
        qCard = Query_card()
        qCard.warp = cards[last_y:last_y+18,:]
        
        # 從card中取出rank_img存入card的屬性中
        Qrank = qCard.warp[:14,2:10]
        Qrank = cv2.cvtColor(Qrank,cv2.COLOR_BGR2GRAY)
        corner_zoom = cv2.resize(Qrank, (0,0), fx=4, fy=4)
        corner_blur = cv2.GaussianBlur(corner_zoom,(5,5),0)
        Qrank_sized = cv2.resize(corner_blur, (RANK_WIDTH, RANK_HEIGHT), 0, 0)
        qCard.rank_img = Qrank_sized

        # 從card中取出suit_img存入card的屬性中
        Qsuit = qCard.warp[1:14,26:]
        Qsuit = cv2.cvtColor(Qsuit,cv2.COLOR_BGR2GRAY)
        corner_zoom = cv2.resize(Qsuit, (0,0), fx=4, fy=4)
        corner_blur = cv2.GaussianBlur(corner_zoom,(3,3),0)
        Qsuit_sized = cv2.resize(corner_blur, (SUIT_WIDTH, SUIT_HEIGHT), 0, 0)
        qCard.suit_img = Qsuit_sized
        qCard_list.append(qCard)

        # 利用match_card賦予卡片rank_name suit_name...等屬性
        match_card(qCard)
        # 把紀錄過的卡片從手牌中切除
        last_y += 18
        y_card = y_card + 18
        cards_h -= 18
    
    return qCard_list

def match_card(qCard):
    """
    對卡片物件進行點數和花色匹配。
    
    Description:
    此函數將卡片物件 `QCard` 與訓練用的點數和花色圖像進行匹配，以識別卡片的點數和花色。
    它計算點數圖像和花色圖像與訓練用圖像的差異，並找到最佳匹配的點數和花色。
    如果差異小於某個閾值（`RANK_DIFF_MAX` 和 `SUIT_DIFF_MAX`），則將識別為該點數和花色。

    """

    
    train_ranks = load_ranks( path + '/Card_Imgs/')
    train_suits = load_suits( path + '/Card_Imgs/')
    
    best_rank_match_diff = 10000
    best_suit_match_diff = 10000
    best_rank_match_name = "Unknown"
    best_suit_match_name = "Unknown"
    i = 0

    if (len(qCard.rank_img) != 0) and (len(qCard.suit_img) != 0):
        for Trank in train_ranks:

                diff_img = cv2.absdiff(qCard.rank_img, Trank.img)
                rank_diff = int(np.sum(diff_img)/255)
                
                if rank_diff < best_rank_match_diff:
                    best_rank_diff_img = diff_img
                    best_rank_match_diff = rank_diff
                    best_rank_name = Trank.name 

        # Same process with suit images
        for Tsuit in train_suits:
                
                diff_img = cv2.absdiff(qCard.suit_img, Tsuit.img)
                suit_diff = int(np.sum(diff_img)/255)
                
                if suit_diff < best_suit_match_diff:
                    best_suit_diff_img = diff_img
                    best_suit_match_diff = suit_diff
                    best_suit_name = Tsuit.name

    if (best_rank_match_diff < RANK_DIFF_MAX):
        best_rank_match_name = best_rank_name

    if (best_suit_match_diff < SUIT_DIFF_MAX):
        best_suit_match_name = best_suit_name

    qCard.best_rank_match,qCard.best_suit_match,qCard.rank_diff,qCard.suit_diff = \
    best_rank_match_name, best_suit_match_name, best_rank_match_diff, best_suit_match_diff
    return
    
    
def draw_results(image, qCard):
    """Draw the card name, center point, and contour on the camera image."""
    x = qCard.center[0]
    y = qCard.center[1]
    cv2.circle(image,(x,y),5,(255,0,0),-1)

    rank_name = qCard.best_rank_match
    suit_name = qCard.best_suit_match

    # Draw card name twice, so letters have black outline
    cv2.putText(image,(rank_name+' of'),(x-60,y-10),font,1,(0,0,0),3,cv2.LINE_AA)
    cv2.putText(image,(rank_name+' of'),(x-60,y-10),font,1,(50,200,200),2,cv2.LINE_AA)

    cv2.putText(image,suit_name,(x-60,y+25),font,1,(0,0,0),3,cv2.LINE_AA)
    cv2.putText(image,suit_name,(x-60,y+25),font,1,(50,200,200),2,cv2.LINE_AA)
    
    # Can draw difference value for troubleshooting purposes
    # (commented out during normal operation)
    #r_diff = str(qCard.rank_diff)
    #s_diff = str(qCard.suit_diff)
    #cv2.putText(image,r_diff,(x+20,y+30),font,0.5,(0,0,255),1,cv2.LINE_AA)
    #cv2.putText(image,s_diff,(x+20,y+50),font,0.5,(0,0,255),1,cv2.LINE_AA)

    return image

# 去除圖像噪音保留卡牌輪廓
def denoise_image(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h_min = 0
    s_min = 0
    v_min = 230
    h_max = 179
    s_max = 58
    v_max = 255
    lower = np.array([h_min,s_min,v_min])
    upper = np.array([h_max,s_max,v_max])
    mask = cv2.inRange(hsv, lower, upper)
    denoised_image = cv2.bitwise_and(image, image ,mask = mask)
    denoised_image = cv2.cvtColor(denoised_image,cv2.COLOR_BGR2GRAY)
    denoised_image = cv2.GaussianBlur(denoised_image,(1,1),0)
    return denoised_image