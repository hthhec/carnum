# -*- coding: utf-8 -*-
import os
import cv2 #실시간 컴퓨터 비전을 목적으로 한 프로그래밍 라이브러리(opencv)
import numpy as np # 다차원 배열을 쉽게 처리 할 수 있도록 지원하는 파이썬 라이브러리
import matplotlib.pyplot as plt #데이터를 차트나 플롯으로 그려주는 라이브러리 패키지로 가장 많이 사용되는 데이터시각화(Data Visualization) 패키지
import pytesseract # image 에서 텍스트 추출하는 , 오픈소스 OCR 엔진



plt.style.use('dark_background')

#print(os.getcwd()) #현재경로확인
#print(os.path.abspath('.')) #절대경로확인
#img_ori = cv2.imread('D:\회사\스터디\python\carnum\cn_2\license_plate_recognition-master\1.jpg')
img_ori = cv2.imread('1.jpg') #이미지 불러오기


height, width, channel = img_ori.shape #사이즈 값 변수에 넣어줌
#print(height, width, channel)


gray = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY) #BGR을 그레이 칼라로 바꿔줌
plt.figure(figsize=(12, 10)) # 이미지 사이즈
plt.imshow(gray, cmap='gray')  # cmap = colormap, matplot쪽
plt.show() #이미지 띄워줌



######
#adaptive thresholding
img_blurred = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0) # 노이즈 줄이기 위해 가우시안블러 사용
#가우시안 블러 : 이미지를 흐릿하게 하여 부드럽게 처리 -> 노이즈가 줄어드는 효과
#어댑티브 쓰레쉬 홀드 : 
#스레시홀딩이란 여러 값을 어떤 임계점을 기준으로 두 가지 부류로 나누는 방법을 의미
#전역 스레시홀딩이 매번 좋은 성능을 내는 것은 아닙니다. 원본 이미지에서 조명이 일정하지 않거나 배경색이 여러 개인 경우에는 하나의 임계값으로 선명한 바이너리 이미지를 만들어내기 힘들 수도 있습니다. 
#이때는 이미지를 여러 영역으로 나눈 뒤, 그 주변(이웃한 영역의) 픽셀 값만 활용하여 임계값을 구해야 하는데, 이를 적응형 스레시홀딩(Adaptive Thresholding)
#평균값(Adapted-Mean)을 활용한 것이 가우시안 분포(Adapted-Gaussian)을 활용한 것보다 더 선명한데 그만큼 잡티가 조금 있습니다. 반면, 가우시안 분포를 활용한 것은 평균값을 활용한 것에 비해 선명도는 조금 떨어지지만 잡티가 더 적습니다
#전체 이미지에 총 9개의 블록을 설정합니다. 이미지를 9등분 한다고 보시면 됩니다. 그 다음 각 블록별로 임계값을 정합니다. 이때, cv2.ADAPTIVE_THRESH_MEAN_C를 파라미터로 전달하면 각 블록의 이웃 픽셀의 평균으로 임계값을 정합니다. cv2.ADAPTIVE_THRESH_GAUSSIAN_C를 파라미터로 전달하면 가우시안 분포에 따른 가중치의 합으로 임계값을 정합니다. 정해진 임계값을 바탕으로 각 블록별로 스레시홀딩을 합니다. 그렇게 하면 전역 스레시홀딩을 적용한 것보다 더 선명하고 부드러운 결과를 얻을 수 있습니다.
img_thresh = cv2.adaptiveThreshold( # 쓰레쉬홀드를 준다 , 임계값 보다 높으면 255, 아니면 0
    img_blurred, 
    maxValue=255.0, 
    adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, # 가우시안 분포에 따른 가중치의 합으로 결정
    thresholdType=cv2.THRESH_BINARY_INV, 
    blockSize=19, 
    C=9
)
plt.figure(figsize=(12, 10))
plt.imshow(img_thresh, cmap='gray')
plt.show()


#######
#컨투어
contours, _ = cv2.findContours(
    img_thresh, 
    mode=cv2.RETR_LIST, 
    method=cv2.CHAIN_APPROX_SIMPLE
)

temp_result = np.zeros((height, width, channel), dtype=np.uint8)

cv2.drawContours(temp_result, contours=contours, contourIdx=-1, color=(255, 255, 255))

plt.figure(figsize=(12, 10))
plt.imshow(temp_result)
plt.show()

###################################################
#prepare data
temp_result = np.zeros((height, width, channel), dtype=np.uint8)

contours_dict = []

for contour in contours:
    x, y, w, h = cv2.boundingRect(contour) #윤곽선을 감싸는 사각형의 값을 구하는 함수
    cv2.rectangle(temp_result, pt1=(x, y), pt2=(x+w, y+h), color=(255, 255, 255), thickness=2)
    #이미지에 사각형을 그리는 함수

    # insert to dict
    contours_dict.append({
        'contour': contour,
        'x': x,
        'y': y,
        'w': w,
        'h': h,
        'cx': x + (w / 2), #중심좌표
        'cy': y + (h / 2)
    })

plt.figure(figsize=(12, 10))
plt.imshow(temp_result, cmap='gray')
plt.show()

###############################################
#번호판 컨투어를 찾는 과정
MIN_AREA = 80 #최소넓이
MIN_WIDTH, MIN_HEIGHT = 2, 8 #최소 너비, 길이
MIN_RATIO, MAX_RATIO = 0.25, 1.0 #최소, 최대 가로 대비 세로 비율
# 번호판의 형태는 정해져있으니깐

possible_contours = []

cnt = 0
for d in contours_dict:
    area = d['w'] * d['h'] #넓이
    ratio = d['w'] / d['h'] #비율 계산
    
    if area > MIN_AREA \
    and d['w'] > MIN_WIDTH and d['h'] > MIN_HEIGHT \
    and MIN_RATIO < ratio < MAX_RATIO:
        d['idx'] = cnt # 각 윤곽선마다 인덱스 값도 같이 저장
        cnt += 1
        possible_contours.append(d) #번호판인 것 같은 애들만 append
        
# visualize possible contours
temp_result = np.zeros((height, width, channel), dtype=np.uint8)

for d in possible_contours: # 번호판 같은애들만 그린다.
#     cv2.drawContours(temp_result, d['contour'], -1, (255, 255, 255))
    cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), color=(255, 255, 255), thickness=2)

plt.figure(figsize=(12, 10))
plt.imshow(temp_result, cmap='gray')
plt.show()


###1####
######################################################
#possible contour 들을 보고 번호판 일 것 같은 후보군을 찾는 작업
#번호판은 순차적으로 정렬이 되어있다.
#배열의 모양을 보고 추려낸다.
#