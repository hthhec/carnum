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