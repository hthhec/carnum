# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract



plt.style.use('dark_background')

#print(os.getcwd()) #현재경로확인
#print(os.path.abspath('.')) #절대경로확인

img_ori = cv2.imread('1.jpg') #이미지 불러오기
#img_ori = cv2.imread('D:\회사\스터디\python\carnum\cn_2\license_plate_recognition-master\1.jpg')

height, width, channel = img_ori.shape #사이즈 값 변수에 넣어줌
#print(height, width, channel)

gray = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY) #그레이 칼라로 바꿔줌
#plt.figure(figsize=(12, 10)) # 이미지 사이즈
#plt.imshow(gray, cmap='gray')  # cmap이 뭔지 알아보기

#plt.imshow(img_ori)
#plt.show() #이미지 띄워줌



######
#어댑티브쓰레쉬 홀딩
img_blurred = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0) # 노이즈 줄이기 위해 가우시안블러 사용
#가우시안 블러에 대하여..
#어댑티브 쓰레쉬 홀드에 대하여
img_thresh = cv2.adaptiveThreshold( # 쓰레쉬홀드를 준다 , 임계값 보다 높으면 255, 아니면 0
    img_blurred, 
    maxValue=255.0, 
    adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
    thresholdType=cv2.THRESH_BINARY_INV, 
    blockSize=19, 
    C=9
)


# plt.figure(figsize=(12, 10))
# plt.imshow(img_thresh, cmap='gray')
#plt.show()



#######
#컨투어

contours, _ = cv2.findContours(
    img_thresh, 
    mode=cv2.RETR_LIST, 
    method=cv2.CHAIN_APPROX_SIMPLE
)

temp_result = np.zeros((height, width, channel), dtype=np.uint8)

cv2.drawContours(temp_result, contours=contours, contourIdx=-1, color=(255, 255, 255))

# plt.figure(figsize=(12, 10))
# plt.imshow(temp_result)
# plt.show()

###################################################
temp_result = np.zeros((height, width, channel), dtype=np.uint8)

contours_dict = []

for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(temp_result, pt1=(x, y), pt2=(x+w, y+h), color=(255, 255, 255), thickness=2)
    
    # insert to dict
    contours_dict.append({
        'contour': contour,
        'x': x,
        'y': y,
        'w': w,
        'h': h,
        'cx': x + (w / 2),
        'cy': y + (h / 2)
    })

# plt.figure(figsize=(12, 10))
# plt.imshow(temp_result, cmap='gray')
# plt.show()

###############################################
MIN_AREA = 80
MIN_WIDTH, MIN_HEIGHT = 2, 8
MIN_RATIO, MAX_RATIO = 0.25, 1.0

possible_contours = []

cnt = 0
for d in contours_dict:
    area = d['w'] * d['h']
    ratio = d['w'] / d['h']
    
    if area > MIN_AREA \
    and d['w'] > MIN_WIDTH and d['h'] > MIN_HEIGHT \
    and MIN_RATIO < ratio < MAX_RATIO:
        d['idx'] = cnt
        cnt += 1
        possible_contours.append(d)
        
# visualize possible contours
temp_result = np.zeros((height, width, channel), dtype=np.uint8)

for d in possible_contours:
#     cv2.drawContours(temp_result, d['contour'], -1, (255, 255, 255))
    cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), color=(255, 255, 255), thickness=2)

plt.figure(figsize=(12, 10))
plt.imshow(temp_result, cmap='gray')
plt.show()