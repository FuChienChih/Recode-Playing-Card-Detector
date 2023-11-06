from PIL import ImageGrab
import numpy as np
import cv2
import Cards


RANK_WIDTH = 38
# 一開始不偵測卡牌
s_pressed = False
serial_number_dict = {}
# 定義捕捉域坐标
bbox = (0, 60, 1440, 780)
image = ImageGrab.grab(bbox=bbox)
width, height = image.size
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter('test.avi', fourcc, 25, (width, height))
case_action = 0 # 0:發牌中 1:結算中 2:洗牌中 3:其他
while True:
    # 捕捉指定區域每貞內容
    img_rgb = ImageGrab.grab(bbox=bbox)
    img_bgr = cv2.cvtColor(np.array(img_rgb), cv2.COLOR_RGB2BGR)
    # img_bgr = cv2.resize(img_bgr, (1980, 988), interpolation=cv2.INTER_CUBIC)
    video.write(img_bgr)
    # cv2.resize(img_bgr)
    # 原始畫面為org_image / 降噪後denoise_image
    org_image = img_bgr
    denoise_image = Cards.denoise_image(img_bgr)

    # 若按下s，s_pressed = True開始偵測卡牌
    if cv2.waitKey(1) & 0xFF == ord('s'):
        if s_pressed == False:
            s_pressed = True
            print('開始偵測')
        else:
            s_pressed = False
            print('暫停偵測')
    

    # 如果偵測動作為“發牌中”
    if case_action == 0:
        # 偵測輪廓，若無不執行
        cnts_sort, cnt_is_card = Cards.find_cards(denoise_image)
        if len(cnts_sort) != 0 and s_pressed:
        # 從所有輪廓中找到牌，並將影像存入 cards
            for i in range(len(cnts_sort)):
                if cnt_is_card[i] == 1:
                    # 判斷card的size決定要不要再分割
                    x,y,w,h = cv2.boundingRect(cnts_sort[i])
                    # print(x,y,w,h)
                    # if w == 33:
                        # print('==============\n',cnts_sort[i],'\n==============') 
                    # if x == 975:
                        # print('==============\n',cnts_sort[i],'\n==============') 
                    # 將牌至中
                    w-=2
                    x+=1
                    y+=1
                    
                    if  w < 45:
                        # 判斷手牌是否被紀錄過
                        min_distance = 3
                        print(x,y,w,h)
                        if not all(abs(x - existing_key) > min_distance for existing_key in serial_number_dict) \
                        and x not in serial_number_dict:
                            pass
                        elif x in serial_number_dict:
                            if h > serial_number_dict[x][0] and h -  serial_number_dict[x][0]>3:
                                serial_number_dict[x][0] = h
                                serial_number_dict[x][1] = Cards.preprocess_card(x,y,w,h,org_image)
                                Cards.update_data(x,serial_number_dict,'temp_data.csv',new_hand = False)
                            else:
                                pass
                        else:
                            serial_number_dict[x] = [h,Cards.preprocess_card(x,y,w,h,org_image)]
                            Cards.update_data(x,serial_number_dict,'temp_data.csv',new_hand = True)
    # 如果偵測動作為“結算中”
    elif case_action == 1:
        pass
    # 如果偵測動作為“洗牌中”
    elif case_action == 2:
        pass
    
    # 顯示偵測畫面
    cv2.imshow('imm', img_bgr)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
print(len(serial_number_dict))
print(serial_number_dict)

video.release()
cv2.destroyAllWindows()
for i in serial_number_dict:
    for j in serial_number_dict[i][1]:
        cv2.imshow('???', j.warp)
        cv2.waitKey(0)