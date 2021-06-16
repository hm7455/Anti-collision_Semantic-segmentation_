# -*- coding: utf-8 -*-
import cv2
import numpy as np
import operator
from functools import reduce

# 读取视频
size = (1024, 1024)
cap = cv2.VideoCapture("/home/xuekelou414/hm/Semantic-Segmentation-Suite-master/video_downloads/outputVideo.avi")
video_writer = cv2.VideoWriter(r'1.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, size)

while cap.isOpened():
    ret, frame = cap.read()
    if cv2.waitKey(100) & 0xff == ord('q'):
        break
    # cv2.imshow('frame',frame)

    img = frame
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换颜色空间
    ret, thresh = cv2.threshold(gray, 5, 100, 0)
    contours, hier = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    img = cv2.drawContours(img, contours, -1, (128, 255, 255), 1)
    image = []
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        a = contours[i]
        tem1 = 0
        num1 = len(a)
        b = a[:, 0]
        if area < 60000 or max(contours[i][:, 0][:, 0]) - min(contours[i][:, 0][:, 0]) < 400:
            for j in range(num1):
                c = b[j]
                cv2.drawContours(img, contours, i, (0, 0, 0), cv2.FILLED)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Set threshold level
    threshold_green = 75
    threshold_red = 38
    # Find coordinates of all pixels below threshold
    coords_green = np.column_stack(np.where(gray == threshold_green))
    coords_red = np.column_stack(np.where(gray == threshold_red))

    # stop the frame
    if len(coords_green) < 30000:
        img = cv2.putText(img, "security: safe", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (250, 0, 0), 2)

        # exit(0)    #image
        cv2.imshow('result', img)
        video_writer.write(img)
        continue  # video

    # print(coords_green)
    # green upper
    [greenminx_index, greenminy_index] = coords_green.argmin(axis=0)
    greenx1 = coords_green[greenminx_index][0]
    greeny1 = coords_green[greenminx_index][1]

    greenx = greenx1 + 20
    index1 = coords_green[np.where(coords_green[:, 0] < greenx)][:, 1]
    # greeny2=min(index1)+10
    # greeny3=max(index1)-10
    greeny2 = min(index1)
    greeny3 = max(index1)
    greeny4 = (greeny2 + greeny3) / 2

    # green_length_threshold_min=420
    green_length_threshold_max = 700
    green_length_threshold = greeny3 - greeny2

    if greeny3 - greeny2 < 420:
        img = cv2.putText(img, "security: safe", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (250, 0, 0), 2)

        # exit(0)    #image
        cv2.imshow('result', img)
        video_writer.write(img)
        continue  # video

    # red lower
    index = coords_red[np.where(coords_red[:, 1] == greeny1)]

    ju = 'record_red_max1' in locals().keys()

    if len(index) == 0 and ju == False:
        img = cv2.putText(img, "The object is blocked", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
        cv2.imshow('result', img)
        video_writer.write(img)
        continue

    elif len(index) == 0:
        redx1 = record_red_max1 + record_red_length1
        redx1_s = record_red_max1
    else:
        redx1 = index[index.argmax(axis=0)[0]][0]
        redx1_s = index[index.argmin(axis=0)[0]][0]

    index2 = coords_red[np.where(coords_red[:, 1] == greeny2)]

    if len(index2) == 0 and ju == False:
        img = cv2.putText(img, "The object is blocked", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
        cv2.imshow('result', img)
        video_writer.write(img)
        continue
    elif len(index2) == 0:
        redx2 = record_red_max2 + record_red_length2
        redx2_s = record_red_max2
    else:
        redx2 = index2[index2.argmax(axis=0)[0]][0]
        redx2_s = index2[index2.argmin(axis=0)[0]][0]
    index5 = coords_green[np.where(coords_green[:, 1] == greeny2)]
    greenx2 = index5[index5.argmin(axis=0)[0]][0]

    index3 = coords_red[np.where(coords_red[:, 1] == greeny3)]
    if len(index3) == 0 and ju == False:
        img = cv2.putText(img, "The object is blocked", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
        cv2.imshow('result', img)
        video_writer.write(img)
        continue
    elif len(index3) == 0:
        redx3 = record_red_max3 + record_red_length3
        redx3_s = record_red_max3
    else:
        redx3 = index3[index3.argmax(axis=0)[0]][0]
        redx3_s = index3[index3.argmin(axis=0)[0]][0]
    index6 = coords_green[np.where(coords_green[:, 1] == greeny3)]
    greenx3 = index6[index6.argmin(axis=0)[0]][0]

    index4 = coords_red[np.where(coords_red[:, 1] == greeny4)]
    if len(index4) == 0 and ju == False:
        img = cv2.putText(img, "The object is blocked", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
        cv2.imshow('result', img)
        video_writer.write(img)
        continue
    elif len(index4) == 0:
        redx4 = record_red_max4 + record_red_length4
        redx4_s = record_red_max4
    else:
        redx4 = index4[index4.argmax(axis=0)[0]][0]
        redx4_s = index4[index4.argmin(axis=0)[0]][0]
    index7 = coords_green[np.where(coords_green[:, 1] == greeny4)]
    greenx4 = index7[index7.argmin(axis=0)[0]][0]

    # record red length  (video)
    # print(greenx1-redx1_s)
    if (greenx1 - redx1_s > 160):
        record_red_length1 = redx1 - redx1_s
        record_red_max1 = redx1_s
        record_red_length2 = redx2 - redx2_s
        record_red_max2 = redx2_s
        record_red_length3 = redx3 - redx3_s
        record_red_max3 = redx3_s
        record_red_length4 = redx4 - redx4_s
        record_red_max4 = redx4_s

    if ju == False:
        record_red_length1 = redx1 - redx1_s
        record_red_max1 = redx1_s
        record_red_length2 = redx2 - redx2_s
        record_red_max2 = redx2_s
        record_red_length3 = redx3 - redx3_s
        record_red_max3 = redx3_s
        record_red_length4 = redx4 - redx4_s
        record_red_max4 = redx4_s

    # image display
    if green_length_threshold < green_length_threshold_max:
        for j in range(greeny1 - 3, greeny1 + 3):
            for x in range(record_red_max1 + record_red_length1, greenx1):
                img[x, j] = (250, 0, 0)
        for j in range(greeny2 - 3, greeny2 + 3):
            for x in range(record_red_max2 + record_red_length2, greenx2):
                img[x, j] = (250, 0, 0)
        for j in range(greeny3 - 3, greeny3 + 3):
            for x in range(record_red_max3 + record_red_length3, greenx3):
                img[x, j] = (250, 0, 0)
        for j in range(greeny4 - 3, greeny4 + 3):
            for x in range(record_red_max4 + record_red_length4, greenx4):
                img[x, j] = (250, 0, 0)
    else:
        img = cv2.putText(img, "security: safe", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (250, 0, 0), 2)

        # exit(0)  #image
        cv2.imshow('result', img)
        video_writer.write(img)

        continue  # video

    mindistance = min(greenx1 - record_red_max1 - record_red_length1, greenx2 - record_red_max2 - record_red_length2,
                      greenx3 - record_red_max3 - record_red_length3, greenx4 - record_red_max4 - record_red_length4)

    img = cv2.putText(img, "distance: " + str(mindistance), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (250, 0, 0), 2)

    # if record_red_max-redx1_s>20:
    #   img = cv2.putText(image, "security: unsafe", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (250, 0, 0), 2)
    if green_length_threshold > green_length_threshold_max:
        img = cv2.putText(img, "security: safe", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (250, 0, 0), 2)
    elif mindistance > 10:
        img = cv2.putText(img, "security: safe", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (250, 0, 0), 2)
    else:
        img = cv2.putText(img, "security: stop rising", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
    cv2.imshow('result', img)
    video_writer.write(img)
video_writer.release()
#
# cv2.imshow('result',img)
# cv2.imwrite("D:/3.png", img)
cv2.waitKey(0)