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
    """Structure to store information about query cards in the camera image."""

    def __init__(self):
        self.contour = [] # Contour of card
        self.x, self.y = 0, 0
        self.width, self.height = 0, 0 # Width and height of card
        self.corner_pts = [] # Corner points of card
        self.center = [] # Center point of card
        self.warp = [] # 200x300, flattened, grayed, blurred image
        self.rank_img = [] # Thresholded, sized image of card's rank
        self.suit_img = [] # Thresholded, sized image of card's suit
        self.best_rank_match = "Unknown" # Best matched rank
        self.best_suit_match = "Unknown" # Best matched suit
        self.rank_diff = 0 # Difference between rank image and best matched train rank image
        self.suit_diff = 0 # Difference between suit image and best matched train suit image

class Train_ranks:
    """Structure to store information about train rank images."""

    def __init__(self):
        self.img = [] # Thresholded, sized rank image loaded from hard drive
        self.name = "Placeholder"

class Train_suits:
    """Structure to store information about train suit images."""

    def __init__(self):
        self.img = [] # Thresholded, sized suit image loaded from hard drive
        self.name = "Placeholder"

### Functions ###
def load_ranks(filepath):
    """Loads rank images from directory specified by filepath. Stores
    them in a list of Train_ranks objects."""

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
    """Loads suit images from directory specified by filepath. Stores
    them in a list of Train_suits objects."""

    train_suits = []
    i = 0
    
    for Suit in ['Spades','Diamonds','Clubs','Hearts']:
        train_suits.append(Train_suits())
        train_suits[i].name = Suit
        filename = Suit + '.jpg'
        train_suits[i].img = cv2.imread(filepath+filename, cv2.IMREAD_GRAYSCALE)
        i = i + 1

    return train_suits

def preprocess_image(image):
    """Returns a grayed, blurred, and adaptively thresholded camera image."""

    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(1,1),0)

    # The best threshold level depends on the ambient lighting conditions.
    # For bright lighting, a high threshold must be used to isolate the cards
    # from the background. For dim lighting, a low threshold must be used.
    # To make the card detector independent of lighting conditions, the
    # following adaptive threshold method is used.
    #
    # A background pixel in the center top of the image is sampled to determine
    # its intensity. The adaptive threshold is set at 50 (THRESH_ADDER) higher
    # than that. This allows the threshold to adapt to the lighting conditions.
    img_w, img_h = np.shape(image)[:2]
    bkg_level = gray[int(img_h/100)][int(img_w/2)]
    thresh_level = bkg_level + BKG_THRESH

    retval, thresh = cv2.threshold(blur,thresh_level,255,cv2.THRESH_BINARY)
    
    return blur

def find_cards(thresh_image):
    """Finds all card-sized contours in a thresholded camera image.
    Returns the number of cards, and a list of card contours sorted
    from largest to smallest."""

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

    # Determine which of the contours are cards by applying the
    # following criteria: 1) Smaller area than the maximum card size,
    # 2), bigger area than the minimum card size, 3) have no parents,
    # and 4) have four corners

    for i in range(len(cnts_sort)):
        size = cv2.contourArea(cnts_sort[i])
        peri = cv2.arcLength(cnts_sort[i],True) # 是否閉合
        approx = cv2.approxPolyDP(cnts_sort[i],0.02*peri,True) # 方形容錯率0.02
        if ((size < CARD_MAX_AREA) and (size > CARD_MIN_AREA)
            and (hier_sort[i][3] == -1) and (len(approx) == 4)):
            
            cnt_is_card[i] = 1

    return cnts_sort, cnt_is_card

def preprocess_card(x,y,w,h,image):
    """Uses contour to find information about the query card. Isolates rank
    and suit images from the card."""
    qCard_list = []
    cards = image[y:y+h,x:x+w]
    cards_w = 38
    cards_h = int(h*38/w)
    cards = cv2.resize(cards, (cards_w, cards_h),0,0)
    x_card,y_card = x,y
    last_y = 0
    while cards_h > 30 :
        qCard = Query_card()
        qCard.width, qCard.height = 38, 15 # 預設卡片大小
        qCard.x, qCard.y = x_card, y_card
        # print(qCard.x, qCard.y, qCard.width, qCard.height)
        pts = np.array([[[qCard.x,qCard.y]],[[qCard.x+qCard.width,qCard.y]],[[qCard.x+qCard.width,qCard.y+qCard.height]]
                        ,[[qCard.x,qCard.y+qCard.height]]], dtype = np.float32)
        qCard.corner_pts = pts
        average = np.sum(pts, axis=0)/len(pts)
        cent_x = int(average[0][0])
        cent_y = int(average[0][1])
        qCard.center = [cent_x, cent_y]
        # ========================
        qCard.warp = cards[last_y:last_y+18,:]
        # cv2.imshow('qCard.warp',qCard.warp)
        # cv2.waitKey(0)
        
        # 從card中取出rank suit存在card的屬性中
        Qrank = qCard.warp[:14,2:10]

        Qrank = cv2.cvtColor(Qrank,cv2.COLOR_BGR2GRAY)
        corner_zoom = cv2.resize(Qrank, (0,0), fx=4, fy=4)
        corner_blur = cv2.GaussianBlur(corner_zoom,(5,5),0)
        Qrank_sized = cv2.resize(corner_blur, (RANK_WIDTH, RANK_HEIGHT), 0, 0)
        qCard.rank_img = Qrank_sized
        
        Qsuit = qCard.warp[1:14,26:]
        # cv2.imshow('Qsuit',Qsuit)
        # cv2.waitKey(0)
        Qsuit = cv2.cvtColor(Qsuit,cv2.COLOR_BGR2GRAY)
        corner_zoom = cv2.resize(Qsuit, (0,0), fx=4, fy=4)
        corner_blur = cv2.GaussianBlur(corner_zoom,(3,3),0)
        Qsuit_sized = cv2.resize(corner_blur, (SUIT_WIDTH, SUIT_HEIGHT), 0, 0)
        
        qCard.suit_img = Qsuit_sized
        # cv2.imshow('Qsuit_sized',Qsuit_sized)
        # cv2.imshow('qCard.rank_img ',qCard.rank_img )
        # cv2.waitKey(0)
        
        qCard_list.append(qCard)

        last_y += 18
        x_card,y_card = x_card ,y_card + 18
        cards_h -= 18
    
    return qCard_list
    # Initialize new Query_card object
    # qCard = Query_card()
    # pts = np.array([[[x,y]],[[x+w,y]],[[x+w,y+h]],[[x,y+h]]], dtype = np.float32)

    # qCard.corner_pts = pts

    # qCard.width, qCard.height = 38, 15

    # average = np.sum(pts, axis=0)/len(pts)
    # cent_x = int(average[0][0])
    # cent_y = int(average[0][1])
    # qCard.center = [cent_x, cent_y]

    # qCard.warp = flattener(image, pts, w, h)
    # cv2.imshow('card',qCard.warp)
    # cv2.waitKey(0)
    # Qcorner_zoom = cv2.resize(qCard.warp, (0,0), fx=4, fy=4)
    
    # # 從card中取出rank suit存在card的屬性中
    # Qrank = qCard.warp[:,1:10]

    # Qrank = cv2.cvtColor(Qrank,cv2.COLOR_BGR2GRAY)
    # corner_zoom = cv2.resize(Qrank, (0,0), fx=4, fy=4)
    # corner_blur = cv2.GaussianBlur(corner_zoom,(5,5),0)
    # Qrank_sized = cv2.resize(corner_blur, (RANK_WIDTH, RANK_HEIGHT), 0, 0)
    # qCard.rank_img = Qrank_sized
    
    # Qsuit = qCard.warp[1:14,26:]

    # Qsuit = cv2.cvtColor(Qsuit,cv2.COLOR_BGR2GRAY)
    # corner_zoom = cv2.resize(Qsuit, (0,0), fx=4, fy=4)
    # corner_blur = cv2.GaussianBlur(corner_zoom,(3,3),0)
    # Qsuit_sized = cv2.resize(corner_blur, (SUIT_WIDTH, SUIT_HEIGHT), 0, 0)
    
    # qCard.suit_img = Qsuit_sized


    return qCard_list

def match_card(qCard, train_ranks, train_suits):
    """Finds best rank and suit matches for the query card. Differences
    the query card rank and suit images with the train rank and suit images.
    The best match is the rank or suit image that has the least difference."""

    best_rank_match_diff = 10000
    best_suit_match_diff = 10000
    best_rank_match_name = "Unknown"
    best_suit_match_name = "Unknown"
    i = 0

    # If no contours were found in query card in preprocess_card function,
    # the img size is zero, so skip the differencing process
    # (card will be left as Unknown)
    if (len(qCard.rank_img) != 0) and (len(qCard.suit_img) != 0):
        # Difference the query card rank image from each of the train rank images,
        # and store the result with the least difference
        for Trank in train_ranks:
                # cv2.imshow('img',Trank.img)
                # cv2.waitKey(0)

                diff_img = cv2.absdiff(qCard.rank_img, Trank.img)
                # cv2.imshow('Qcard_rank',qCard.rank_img)
                # cv2.imshow('TrainCard_rank',Trank.img)
                # cv2.waitKey(0)
                rank_diff = int(np.sum(diff_img)/255)
                # print(f'{ Trank.name}:{rank_diff}')
                
                if rank_diff < best_rank_match_diff:
                    best_rank_diff_img = diff_img
                    best_rank_match_diff = rank_diff
                    best_rank_name = Trank.name 

        # Same process with suit images
        for Tsuit in train_suits:
                
                diff_img = cv2.absdiff(qCard.suit_img, Tsuit.img)
                # cv2.imshow('Qcard_suit',qCard.suit_img)
                # cv2.imshow('TrainCard_suit',Tsuit.img)
                # cv2.waitKey(0)
                suit_diff = int(np.sum(diff_img)/255)
                
                if suit_diff < best_suit_match_diff:
                    best_suit_diff_img = diff_img
                    best_suit_match_diff = suit_diff
                    best_suit_name = Tsuit.name

    # Combine best rank match and best suit match to get query card's identity.
    # If the best matches have too high of a difference value, card identity
    # is still Unknown
    if (best_rank_match_diff < RANK_DIFF_MAX):
        best_rank_match_name = best_rank_name

    if (best_suit_match_diff < SUIT_DIFF_MAX):
        best_suit_match_name = best_suit_name

    # Return the identiy of the card and the quality of the suit and rank match
    return best_rank_match_name, best_suit_match_name, best_rank_match_diff, best_suit_match_diff
    
    
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

def flattener(image, pts, w, h):
    """Flattens an image of a card into a top-down 200x300 perspective.
    Returns the flattened, re-sized, grayed image.
    See www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/"""
    temp_rect = np.zeros((4,2), dtype = "float32")
    
    s = np.sum(pts, axis = 2)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]

    diff = np.diff(pts, axis = -1)
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]


    temp_rect[0] = tl
    temp_rect[1] = tr
    temp_rect[2] = br
    temp_rect[3] = bl
            
        
    maxWidth = 38
    maxHeight = 15

    # Create destination array, calculate perspective transform matrix,
    # and warp card image
    dst = np.array([[0,0],[maxWidth-1,0],[maxWidth-1,maxHeight-1],[0, maxHeight-1]], np.float32)
    M = cv2.getPerspectiveTransform(temp_rect,dst)
    warp = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # warp = cv2.cvtColor(warp,cv2.COLOR_BGR2GRAY)

        

    return warp
