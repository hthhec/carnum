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
# plt.figure(figsize=(12, 10)) # 이미지 사이즈
# plt.imshow(gray, cmap='gray')  # cmap = colormap, matplot쪽
# plt.show() #이미지 띄워줌



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
# plt.figure(figsize=(12, 10))
# plt.imshow(img_thresh, cmap='gray')
# plt.show()


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

# plt.figure(figsize=(12, 10))
# plt.imshow(temp_result, cmap='gray')
# plt.show()

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

# plt.figure(figsize=(12, 10))
# plt.imshow(temp_result, cmap='gray')
# plt.show()


###1####
######################################################
#possible contour 들을 보고 번호판 일 것 같은 후보군을 찾는 작업
#번호판은 순차적으로 정렬이 되어있다.
#배열의 모양을 보고 추려낸다.
#


MAX_DIAG_MULTIPLYER = 5 # 5  #컨투어와 컨투어 사이의 길이를 제한, 
# 첫 컨투어 대각선길이(diag_len) 다음 컨투어 길이가 5배 안에 있어야한다.
MAX_ANGLE_DIFF = 12.0 # 12.0 # 첫 컨투어 두번째 컨투어 중심 값을 이었을 때 세타 값을 제한
MAX_AREA_DIFF = 0.5 # 0.5 # 컨투어 "면적"차이 가 너무 많이 나면 노노
MAX_WIDTH_DIFF = 0.8 # 컨투어 "너비"차이 가 너무 많이 나면 노노
MAX_HEIGHT_DIFF = 0.2 # 컨투어 "높이"차이 가 너무 많이 나면 노노
MIN_N_MATCHED = 3 # 3  # 위의 조건들이 3개 이상 만족해야 번호판이다. 

def find_chars(contour_list): #재귀함수로 계속 돌릴 함수
    matched_result_idx = [] #여기에 최종 인덱스 결과값들을 저장할거다.
    
    for d1 in contour_list:  # d1 과 d2 를 비교
        matched_contours_idx = []
        for d2 in contour_list:
            if d1['idx'] == d2['idx']: 
                continue #같으면 비교할 필요가 없으니깐 컨티뉴로 넘긴다.

            dx = abs(d1['cx'] - d2['cx']) #
            dy = abs(d1['cy'] - d2['cy'])

            diagonal_length1 = np.sqrt(d1['w'] ** 2 + d1['h'] ** 2)

            distance = np.linalg.norm(np.array([d1['cx'], d1['cy']]) - np.array([d2['cx'], d2['cy']])) # 컨투어 거리를 구한다 (대각선)
            if dx == 0:
                angle_diff = 90  #x 값이 똑같으면 90도 , 그리고 dx(분모가) 0이면 에러니깐 이렇게 예외처리
            else:
                angle_diff = np.degrees(np.arctan(dy / dx)) # 세타 값 구하기 위해서 
            area_diff = abs(d1['w'] * d1['h'] - d2['w'] * d2['h']) / (d1['w'] * d1['h'])
            width_diff = abs(d1['w'] - d2['w']) / d1['w']
            height_diff = abs(d1['h'] - d2['h']) / d1['h']
            # 비율들을 구한다.

            if distance < diagonal_length1 * MAX_DIAG_MULTIPLYER \
            and angle_diff < MAX_ANGLE_DIFF and area_diff < MAX_AREA_DIFF \
            and width_diff < MAX_WIDTH_DIFF and height_diff < MAX_HEIGHT_DIFF:
                matched_contours_idx.append(d2['idx'])#기준에 맞는 애들만 넣어준다. 

        # append this contour
        matched_contours_idx.append(d1['idx']) #마지막에 d2도 넣어준다.

        if len(matched_contours_idx) < MIN_N_MATCHED:
            continue #3보다 낮으면 컨티뉴해서 그냥 제외 처리

        matched_result_idx.append(matched_contours_idx) # 최종후보군

        unmatched_contour_idx = []  #아닌애들끼리 한번더 비교해준다.
        for d4 in contour_list:
            if d4['idx'] not in matched_contours_idx:
                unmatched_contour_idx.append(d4['idx'])

        unmatched_contour = np.take(possible_contours, unmatched_contour_idx)
        
        # recursive
        recursive_contour_list = find_chars(unmatched_contour) #재귀함수로 돌린다. 리스트가 나온다.
        
        for idx in recursive_contour_list:
            matched_result_idx.append(idx) # 살아남은 애들을 여기 넣어준다.

        break

    return matched_result_idx
    
result_idx = find_chars(possible_contours)

matched_result = []
for idx_list in result_idx:
    matched_result.append(np.take(possible_contours, idx_list))

# visualize possible contours
temp_result = np.zeros((height, width, channel), dtype=np.uint8)

for r in matched_result:
    for d in r:
#         cv2.drawContours(temp_result, d['contour'], -1, (255, 255, 255))
        cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), color=(255, 255, 255), thickness=2)
#남은 애들을 그려본다.

plt.figure(figsize=(12, 10))
plt.imshow(temp_result, cmap='gray')
#plt.show()


################################
#삐뚫어진 번호판을 똑바로


PLATE_WIDTH_PADDING = 1.3 # 1.3
PLATE_HEIGHT_PADDING = 1.5 # 1.5
MIN_PLATE_RATIO = 3
MAX_PLATE_RATIO = 10

plate_imgs = []
plate_infos = []

for i, matched_chars in enumerate(matched_result): #최종 result에 대해  loop를 돌린다.
    sorted_chars = sorted(matched_chars, key=lambda x: x['cx']) #순서대로 정렬

    plate_cx = (sorted_chars[0]['cx'] + sorted_chars[-1]['cx']) / 2 #센터를 구한다.
    plate_cy = (sorted_chars[0]['cy'] + sorted_chars[-1]['cy']) / 2
    
    plate_width = (sorted_chars[-1]['x'] + sorted_chars[-1]['w'] - sorted_chars[0]['x']) * PLATE_WIDTH_PADDING
    
    sum_height = 0
    for d in sorted_chars:
        sum_height += d['h']

    plate_height = int(sum_height / len(sorted_chars) * PLATE_HEIGHT_PADDING)
    
    triangle_height = sorted_chars[-1]['cy'] - sorted_chars[0]['cy']
    triangle_hypotenus = np.linalg.norm(
        np.array([sorted_chars[0]['cx'], sorted_chars[0]['cy']]) - 
        np.array([sorted_chars[-1]['cx'], sorted_chars[-1]['cy']])
    )
    
    angle = np.degrees(np.arcsin(triangle_height / triangle_hypotenus))
    
    rotation_matrix = cv2.getRotationMatrix2D(center=(plate_cx, plate_cy), angle=angle, scale=1.0)
    #opencv 로테이션 매트릭스를 구한다.
    img_rotated = cv2.warpAffine(img_thresh, M=rotation_matrix, dsize=(width, height))
    # 삐뚫어진거를 똑바로 이미지를 변형한다.
    img_cropped = cv2.getRectSubPix( #회전된 이미지에서 원하는 부분만 자른다.
        img_rotated, 
        patchSize=(int(plate_width), int(plate_height)), 
        center=(int(plate_cx), int(plate_cy))
    )
    
    if img_cropped.shape[1] / img_cropped.shape[0] < MIN_PLATE_RATIO or img_cropped.shape[1] / img_cropped.shape[0] < MIN_PLATE_RATIO > MAX_PLATE_RATIO:
        continue
    
    plate_imgs.append(img_cropped)
    plate_infos.append({
        'x': int(plate_cx - plate_width / 2),
        'y': int(plate_cy - plate_height / 2),
        'w': int(plate_width),
        'h': int(plate_height)
    })
    
    plt.subplot(len(matched_result), 1, i+1)
    plt.imshow(img_cropped, cmap='gray')
    #plt.show()


    ##########################################################
    #한번더 쓰레쉬 홀딩

    longest_idx, longest_text = -1, 0
plate_chars = []

for i, plate_img in enumerate(plate_imgs):
    plate_img = cv2.resize(plate_img, dsize=(0, 0), fx=1.6, fy=1.6)
    _, plate_img = cv2.threshold(plate_img, thresh=0.0, maxval=255.0, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #otsu 쓰레쉬 홀드라고 간단한 쓰레쉬 홀딩
    #한번 쓰레쉬 홀딩 된거니깐

    # find contours again (same as above)
    contours, _ = cv2.findContours(plate_img, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)
    #한번더 컨투어를 찾는다.
    #확실하게 번호판이 맞는지 확인
    plate_min_x, plate_min_y = plate_img.shape[1], plate_img.shape[0]
    plate_max_x, plate_max_y = 0, 0

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        #컨투어들의 boundingrect를 구하고
        
        area = w * h
        ratio = w / h

        if area > MIN_AREA \
        and w > MIN_WIDTH and h > MIN_HEIGHT \
        and MIN_RATIO < ratio < MAX_RATIO:
        # 한번 더 체크하고
            if x < plate_min_x:
                plate_min_x = x
            if y < plate_min_y:
                plate_min_y = y
            if x + w > plate_max_x:
                plate_max_x = x + w
            if y + h > plate_max_y:
                plate_max_y = y + h
                # 번호판의 최대 최소 x,y 를 구한다.
                
    img_result = plate_img[plate_min_y:plate_max_y, plate_min_x:plate_max_x]
    
    #글씨 읽기 전에 잘 읽게하려고
    #가우시안 블러 한번더
    #흐려졌으니깐 쓰레쉬홀드도 한번더 
    #쓰레쉬홀드한 이미지에 약간의 패딩(검정의 여백)을 줘서
    img_result = cv2.GaussianBlur(img_result, ksize=(3, 3), sigmaX=0)
    _, img_result = cv2.threshold(img_result, thresh=0.0, maxval=255.0, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    img_result = cv2.copyMakeBorder(img_result, top=10, bottom=10, left=10, right=10, borderType=cv2.BORDER_CONSTANT, value=(0,0,0))

    chars = pytesseract.image_to_string(img_result, lang='kor', config='--psm 7 --oem 0')
    #pytesseract에서 잘 읽을 수 있도록 한다.
    #언어는 kor
    #psm 7 이미지 안에 언어가 한줄이다라는 고정 조건
    #oem 0  은 0번 엔진을 쓴다는 의미, 
    #요즘 엔진은 문맥 파악등 기능 많은데 , 그냥 예전 엔진이 번호판 인식에는 가장 잘 된다.

    result_chars = ''
    has_digit = False
    for c in chars:
        if ord('가') <= ord(c) <= ord('힣') or c.isdigit(): #한글이 있는지
            if c.isdigit():
                has_digit = True #숫자가 하나라도 있는지.
            result_chars += c
    
    print(result_chars)
    plate_chars.append(result_chars)

    if has_digit and len(result_chars) > longest_text:
        longest_idx = i #가장 긴 것을 우리가 찾은 번호파이다 라고 넣어준다.

    plt.subplot(len(plate_imgs), 1, i+1)
    plt.imshow(img_result, cmap='gray')
    #plt.show()


#######################################################################
#최종 위치와 예측한 결과를 보여준다.

info = plate_infos[longest_idx]
chars = plate_chars[longest_idx]

print(chars)

img_out = img_ori.copy()

cv2.rectangle(img_out, pt1=(info['x'], info['y']), pt2=(info['x']+info['w'], info['y']+info['h']), color=(255,0,0), thickness=2)

cv2.imwrite(chars + '.jpg', img_out)

plt.figure(figsize=(12, 10))
plt.imshow(img_out)
plt.show()