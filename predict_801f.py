import os, time, cv2, sys, math
import tensorflow as tf
import argparse
import numpy as np

from tkinter import messagebox
from tkinter import *
import time
import random

from utils import utils, helpers
from builders import model_builder
import threading
# from mttkinter import mtTkinter as tk


global outsignal
global urlgl
global dissss

f = open('distance.txt')
dissss = f.read().strip().split(',')
f.close


f = open('in.txt')
urlgl = f.read().strip().split(',')
f.close
outsignal = np.zeros((81,2))

def savetxt_compact(fname, x, fmt="%d", delimiter=','):
    with open(fname, 'w+') as fh:
        for row in x:
            line = delimiter.join("0" if value == 0 else fmt % value for value in row)
            fh.write(line + '\n')

def callback():
    messagebox.showwarning('警告', '请输入密码')
root = Tk()
root.title("software permission has expired")
root.geometry('400x100')
root.protocol("WM_DELETE_WINDOW", callback)
l1 = Label(root, text="password:")
l1.pack()
xls_text = StringVar()
xls = Entry(root, textvariable=xls_text, show='*')
xls_text.set("")
xls.pack()

dict1 = {'0': 'H', '1': '*', '2': 'q', '3': 'M', '4': '&', '5': 'd', '6': 'W', '7': 'K', '8': 'x', '9': '#'}
dict2 = {'0': 'n', '1': 'B', '2': 'c', '3': 'k', '4': 'F', '5': 'j', '6': '^', '7': 'L', '8': 's', '9': '%'}
dict3 = {'0': 'R', '1': 'r', '2': 'S', '3': '$', '4': 'Y', '5': 'Z', '6': 'A', '7': 'a', '8': 'U', '9': 'i'}


def on_click():
    x = xls_text.get()
    if x == '420420':  # password
        messagebox.showinfo(title='', message='succeed ,please restart')
        stoptime = 2000000
        newx = ''
        for i in range(len(str(stoptime))):
            randdict = random.randint(1, 3)
            if randdict == 1:
                newx += str(dict1[str(stoptime)[i]])
            if randdict == 2:
                newx += str(dict2[str(stoptime)[i]])
            if randdict == 3:
                newx += str(dict3[str(stoptime)[i]])
        f = open("sys.xkl", "w")
        print(newx, file=f)
        f.close()
        sys.exit()
    else:
        messagebox.showinfo(title='', message='password error')
        sys.exit()

stoptime = open('sys.xkl').read().strip()
recotime = ''
for i in range(len(str(stoptime))):
    try:
        recotime += list(dict1.keys())[list(dict1.values()).index(stoptime[i])]
    except:
        try:
            recotime += list(dict2.keys())[list(dict2.values()).index(stoptime[i])]
        except:
            recotime += list(dict3.keys())[list(dict3.values()).index(stoptime[i])]

if int(recotime) <= 0:
    x = Button(root, text="press", command=on_click).pack()

    root.mainloop()

class_names_list, label_values = helpers.get_label_info(os.path.join('./dataset', "class_dict.csv"))

num_classes = len(label_values)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

net_input = tf.placeholder(tf.float32, shape=[None, None, None, 3])
net_output = tf.placeholder(tf.float32, shape=[None, None, None, num_classes])

network, _ = model_builder.build_model(model_name="BiSeNet", net_input=net_input,
                                       num_classes=num_classes,
                                       crop_width=1024,
                                       crop_height=1024,
                                       is_training=False)

sess.run(tf.global_variables_initializer())

print('Loading model checkpoint weights')
saver = tf.train.Saver(max_to_keep=1000)
k = open('checkpoints.txt')
m = k.read()
print(m)
saver.restore(sess, m)


# f = open('in.txt')
# url = f.read().rstrip()
# print(url)
# # url = 'rtsp://admin:Abc12345@192.168.1.69:554/MPEG-4'
# # print(url)
# cap = cv2.VideoCapture(url)

frame_global1 = np.zeros((1024,1024,3),np.uint8)
frame_global2 = np.zeros((1024,1024,3),np.uint8)
frame_global3 = np.zeros((1024,1024,3),np.uint8)
frame_global4 = np.zeros((1024,1024,3),np.uint8)
global comnum1
global comnum2
global comnum3
global comnum4
global k1
global k2
global k3
global k4
global jured1
global jured2
global jured3
global jured4
global mindistance1
global mindistance2
global mindistance3
global mindistance4

mindistance1=-1
mindistance2=-1
mindistance3=-1
mindistance4=-1

comnum1 = -1
comnum2 = -1
comnum3 = -1
comnum4 = -1
jured1=0
jured2=0
jured3=0
jured4=0
k1=-1
k2=-1
k3=-1
k4=-1


def readframe1():
    global frame_global1
    global outsignal
    global urlgl
    global comnum1
    global k1
    global jured1
    global mindistance1
    while True:
        try:
            k1 = -1
            # start_time1 = time.clock()
            #time.sleep(0.03)
            if cv2.waitKey(100) & 0xff == ord('q'):
                break
            size = (1024, 1024)
            # opencv支持不同的编码格式
            # video_writer = cv2.VideoWriter('outputVideo1.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, size)
            # f = open('in.txt')
            # urlgl = f.read().rstrip().split(',')
            url = urlgl
            # start_time = time.clock()

            
            frame_global1=np.zeros((1024,1024,3),np.uint8)
            
            if url[0]=='0':
                continue
            else:
                urlx = 'rtsp://admin:Abc12345@192.168.1.' + str(
                    int(url[0]) + 10) + ':554/MPEG-4 max-buffers=1 drop=true'
            # end_time = time.clock()
            # print('读取RTSP：',end_time - start_time)
            outsignal[comnum1][0] = 0
            outsignal[comnum1][1] = 0
            print('一号：', urlx)
            cap = cv2.VideoCapture(urlx)
            c = 0
            k1 = -1  # fixed the measurement bug when changinging videos
            # timeF = 10
            # jured = 0
            # end_time1 = time.clock()
            # print('while()到cap初始：',end_time1 - start_time1)
            print('1读取信号完毕')
            jured1=0
            mindistance1=-1
            while cap.isOpened():
                start_time = time.clock()
                m = urlgl

                if m[0] != url[0]:
                    outsignal[comnum1][0] = 0
                    outsignal[comnum1][1] = 0
                    break
                
                if isinstance(int(m[0]),int):
                    comnum1 = int(m[0])-1
                else:
                    print("in中存在非数字")
                ret, frame_global1 = cap.read()
        except:
            continue



def readframe2():
    global frame_global2
    global outsignal
    global urlgl
    global comnum2
    global k2
    global jured2
    global mindistance2
    while True:
        try:
            k2 = -1
            # start_time1 = time.clock()
            #time.sleep(0.03)
            if cv2.waitKey(100) & 0xff == ord('q'):
                break
            # try:
            size = (1024, 1024)
            # opencv支持不同的编码格式
            # video_writer = cv2.VideoWriter('outputVideo1.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, size)
            # f = open('in.txt')
            # urlgl = f.read().rstrip().split(',')
            url = urlgl
            # start_time = time.clock()

            frame_global2=np.zeros((1024,1024,3),np.uint8)
            
            if url[1]=='0':
                continue
            else:
                urlx = 'rtsp://admin:Abc12345@192.168.1.' + str(
                    int(url[1]) + 10) + ':554/MPEG-4 max-buffers=1 drop=true'
            # end_time = time.clock()
            # print('读取RTSP：',end_time - start_time)
            outsignal[comnum2][0] = 0
            outsignal[comnum2][1] = 0
            print('二号：', urlx)
            cap = cv2.VideoCapture(urlx)
            c = 0
            # kkk = -1  # fixed the measurement bug when changinging videos
            # timeF = 10
            # jured = 0
            # end_time1 = time.clock()
            # print('while()到cap初始：',end_time1 - start_time1)
            print('2读取信号完毕')
            jured2=0
            mindistance2=-1
            while cap.isOpened():
                start_time = time.clock()
                m = urlgl

                if m[1] != url[1]:
                    outsignal[comnum2][0] = 0
                    outsignal[comnum2][1] = 0
                    break
                
                if isinstance(int(m[1]),int):
                    comnum2 = int(m[1])-1
                else:
                    print("in中存在非数字")
                ret, frame_global2 = cap.read()
        except:
            continue



def readframe3():
    global frame_global3
    global outsignal
    global urlgl
    global comnum3
    global k3
    global jured3
    global mindistance3
    while True:
        try:
            k3 = -1
            # start_time1 = time.clock()
            #time.sleep(0.03)
            if cv2.waitKey(100) & 0xff == ord('q'):
                break
            # try:
            size = (1024, 1024)
            # opencv支持不同的编码格式
            # video_writer = cv2.VideoWriter('outputVideo1.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, size)
            # f = open('in.txt')
            # urlgl = f.read().rstrip().split(',')
            url = urlgl
            
            #start_time = time.clock()

            
            frame_global3=np.zeros((1024,1024,3),np.uint8)
            
            if url[2]=='0':
                continue
            else:
                urlx = 'rtsp://admin:Abc12345@192.168.1.' + str(
                    int(url[2]) + 10) + ':554/MPEG-4 max-buffers=1 drop=true'
            end_time = time.clock()
            print('三号：',urlx)
            cap = cv2.VideoCapture(urlx)
            c = 0
            # kkk = -1  # fixed the measurement bug when changinging videos
            # timeF = 10
            # jured = 0
            # end_time1 = time.clock()
            # print('while()到cap初始：',end_time1 - start_time1)
            outsignal[comnum3][0] = 0
            outsignal[comnum3][1] = 0
            print('3读取信号完毕')
            jured3=0
            mindistance3=-1
            while cap.isOpened():
                # start_time = time.clock()
                m = urlgl
                if m[2] != url[2]:
                    outsignal[comnum3][0] = 0
                    outsignal[comnum3][1] = 0
                    break
                
                if isinstance(int(m[2]),int):
                    comnum3 = int(m[2])-1
                else:
                    print("in中存在非数字")
                ret, frame_global3 = cap.read()
        except:
            continue


def readframe4():
    global frame_global4
    global outsignal
    global urlgl
    global comnum4
    global k4
    global jured4
    global mindistance4
    while True:
        try:
            k4 = -1
            # start_time1 = time.clock()
            #time.sleep(0.03)
            if cv2.waitKey(100) & 0xff == ord('q'):
                break
            # try:
            size = (1024, 1024)
            # opencv支持不同的编码格式
            # video_writer = cv2.VideoWriter('outputVideo1.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, size)
            # f = open('in.txt')
            # urlgl = f.read().rstrip().split(',')
            url = urlgl
            
            # start_time = time.clock()

            
            frame_global4=np.zeros((1024,1024,3),np.uint8)
            
            if url[3]=='0':
                continue
            else:
                urlx = 'rtsp://admin:Abc12345@192.168.1.' + str(
                    int(url[3]) + 10) + ':554/MPEG-4 max-buffers=1 drop=true'
            # end_time = time.clock()
            # print('读取RTSP：',end_time - start_time)
            print('四号：',urlx)
            cap = cv2.VideoCapture(urlx)
            c = 0
            # kkk = -1  # fixed the measurement bug when changinging videos
            # timeF = 10
            # jured = 0
            # end_time1 = time.clock()
            outsignal[comnum4][0] = 0
            outsignal[comnum4][1] = 0
            print('4读取信号完毕')
            jured4=0
            mindistance4=-1
            while cap.isOpened():
                # start_time = time.clock()
                m = urlgl

                if m[3] != url[3]:
                    outsignal[comnum4][0] = 0
                    outsignal[comnum4][1] = 0
                    break
                
                if isinstance(int(m[3]),int):
                    comnum4 = int(m[3])-1
                else:
                    print("in中存在非数字")
                ret, frame_global4 = cap.read()
        except:
            continue


def ca1():
    global outsignal
    global urlgl
    global frame_global1
    global comnum1
    global dissss
    global k1
    global jured1
    global mindistance1
    while True:
        try:
            if frame_global1.any():
                if cv2.waitKey(100) & 0xff == ord('q'):
                    break
                
                loaded_image = frame_global1
                resized_image = cv2.resize(loaded_image, (1024, 1024))
                input_image = np.expand_dims(np.float32(resized_image[:1024, :1024]), axis=0) / 255.0

                st = time.clock()

                output_image = sess.run(network, feed_dict={net_input: input_image})

                # run_time = time.clock()
                # print("run_time", run_time-st)
                # Graytime = time.clock()
                output_image = np.array(output_image[0, :, :, :])
                output_image = helpers.reverse_one_hot(output_image)

                out_vis_image = helpers.colour_code_segmentation(output_image, label_values)
                findContour = time.clock()
                img = cv2.resize(cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_RGB2BGR), (1024, 1024),
                                 interpolation=cv2.INTER_AREA)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换颜色空间
                ret, thresh = cv2.threshold(gray, 5, 100, 0)
                image, contours, hier = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                # print("找轮廓", time.clock()-findContour)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                bodynum = 0
                greenno = 1
                # f = open("out.txt", "w")
                # print(0, file = f)  #py3
                # #print >> f, 0
                # f.close()
                start_contours1 = time.clock()
                # run_time2gray = time.clock()
                # print('toGraytime',run_time2gray-Graytime)

                for i in range(len(contours)):
                    area = cv2.contourArea(contours[i])
                    cv2.drawContours(img, contours, i, (0, 0, 0), cv2.FILLED)
                    if (area > 60000 and (max(contours[i][:, 0][:, 0]) - min(contours[i][:, 0][:, 0])) > 700 and (
                            max(contours[i][:, 0][:, 1]) - min(contours[i][:, 0][:, 1])) < 300):
                        coords_red = contours[i][:, 0]
                        coords_red_area = area
                        jured1 = 1
                        img = cv2.drawContours(img, contours, i, (128, 255, 255), 3)
                        bodynum += 1

                if jured1 == 0:
                    img = cv2.putText(img, "The object is blocked", (600, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                                      (0, 0, 255), 2)
                    out = cv2.addWeighted(resized_image, 0.5, img, 0.5, 0)
                    cv2.imshow('result1', out)
                    # video_writer.write(out)
                    outsignal[comnum1][0] = 2
                    outsignal[comnum1][1] = -1
                    print(str(comnum1+1)+'号相机画面异常')
                    # end_contours1 = time.clock()
                    # print('contours1', end_contours1 - start_contours1)
                    continue
                # start_contours2 = time.clock()
                for i in range(len(contours)):
                    area = cv2.contourArea(contours[i])
                    if (area > 60000 and max(contours[i][:, 0][:, 0]) - min(contours[i][:, 0][:, 0]) > 420 and (
                            max(contours[i][:, 0][:, 1]) - min(contours[i][:, 0][:, 1])) >= 250 and min(
                        contours[i][:, 0][:, 1]) > min(coords_red[:, 1]) + 5):
                        coords_green = contours[i][:, 0]
                        coords_green_area = area
                        greenno = 0
                        bodynum += 1
                        img = cv2.drawContours(img, contours, i, (128, 255, 255), 3)
                        # end_contours2 = time.clock()
                        # print('contours2', end_contours2 - start_contours2)



                if bodynum < 1 and mindistance1<int(dissss[1]) and mindistance1!=-1:
                    img = cv2.putText(img, "security: stop rising", (600, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                                      (0, 255, 255), 2)
                    outsignal[comnum1][0] = 1
                    
                    outsignal[comnum1][1] = 0
                    print(str(comnum1+1)+'号相机画面正常,   距离:  0  距离较近')
                    
                    out = cv2.addWeighted(resized_image, 0.5, img, 0.5, 0)

                    cv2.imshow('result1', out)
                    continue



                greenex = 'coords_green_area' in locals().keys()

                if greenex == False or greenno == 1:
                    img = cv2.putText(img, "security: safe", (600, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (250, 0, 0),
                                      2)
                    img = cv2.putText(img, "distance: far away", (600, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                                      (250, 0, 0), 2)

                    out = cv2.addWeighted(resized_image, 0.5, img, 0.5, 0)
                    cv2.imshow('result1', out)
                    outsignal[comnum1][0] = 0
                    if mindistance1<0:
                        outsignal[comnum1][1] =-1
                        print(str(comnum1+1)+'号相机画面正常,   距离:  >500')
                    else:
                        outsignal[comnum1][1] = mindistance1
                        print(str(comnum1+1)+'号相机画面正常,   距离:  '+str(mindistance1))
                    continue  # video

                [greenminx_index, greenminy_index] = coords_green.argmin(axis=0)
                

                

                greeny1 = coords_green[greenminy_index][0]
                if greeny1<int(dissss[2]) or greeny1>int(dissss[3]):   #  300  600
                    greeny1 = 470

                

                greeny2 = 420
                greeny3 = 580

                greeny4 = 530

                green_length_threshold_max = 700
                green_length_threshold = greeny3 - greeny2

                # if green_length_threshold < 420:
                #     img = cv2.putText(img, "security: safe", (600,100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (250, 0, 0), 2)
                #     img = cv2.putText(img, "distance: far away", (600, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (250, 0, 0), 2)
                #     # exit(0)    #imag
                #     out = cv2.addWeighted(resized_image, 0.5, img, 0.5, 0)
                #     #cv2.imshow('result', out)
                #     # video_writer.write(out)
                #     outsignal[comnum][0]=0
                #     outsignal[comnum][1]=-1

                #     continue  # video

                # red lower
                index = coords_red[np.where(coords_red[:, 0] == greeny1)]

                ju = 'record_red_max1' in locals().keys()
                k1 += 1
                # if len(index) == 0 and ju == False:
                #     img = cv2.putText(img, "The object is blocked", (600, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                #                       (0, 0, 255), 2)
                #     out = cv2.addWeighted(resized_image, 0.5, img, 0.5, 0)
                #     cv2.imshow('result1', out)
                #     # video_writer.write(out)
                #     outsignal[comnum1][0] = 2
                #     outsignal[comnum1][1] = -1
                #     k1-=1
                #     print(str(comnum1+1)+'号相机画面异常')
                #     continue
                if k1 == 0:  # fixed the measurement bug when changinging videos
                    redx1 = index[index.argmax(axis=0)[1]][1]
                    redx1_s = index[index.argmin(axis=0)[1]][1]
                    record_red_length1 = redx1 - redx1_s
                    record_red_max1 = redx1_s
                elif len(index) == 0:
                    redx1 = record_red_max1 + record_red_length1
                    redx1_s = record_red_max1
                else:
                    redx1 = index[index.argmax(axis=0)[1]][1]
                    redx1_s = index[index.argmin(axis=0)[1]][1]
                index2 = coords_red[np.where(coords_red[:, 0] == greeny2)]

                # if len(index2) == 0 and ju == False:
                #     img = cv2.putText(img, "The object is blocked", (600, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                #                       (0, 0, 255), 2)
                #     out = cv2.addWeighted(resized_image, 0.5, img, 0.5, 0)
                #     cv2.imshow('result1', out)
                #     # video_writer.write(out)
                #     outsignal[comnum1][0] = 2
                #     outsignal[comnum1][1] = -1
                #     print(str(comnum1+1)+'号相机画面异常')
                #     continue


                index10=coords_green[np.where(coords_green[:,0]==greeny1)]
                greenx1=index10[index10.argmin(axis=0)[1]][1]
                if k1 == 0:
                    redx2 = index2[index2.argmax(axis=0)[1]][1]
                    redx2_s = index2[index2.argmin(axis=0)[1]][1]
                    record_red_length2 = redx2 - redx2_s
                    record_red_max2 = redx2_s
                elif len(index2) == 0:
                    redx2 = record_red_max2 + record_red_length2
                    redx2_s = record_red_max2
                else:
                    redx2 = index2[index2.argmax(axis=0)[1]][1]
                    redx2_s = index2[index2.argmin(axis=0)[1]][1]
                index5 = coords_green[np.where(coords_green[:, 0] == greeny2)]
                greenx2 = index5[index5.argmin(axis=0)[1]][1]

                index3 = coords_red[np.where(coords_red[:, 0] == greeny3)]
                # if len(index3) == 0 and ju == False:
                #     img = cv2.putText(img, "The object is blocked", (600, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                #                       (0, 0, 255), 2)
                #     out = cv2.addWeighted(resized_image, 0.5, img, 0.5, 0)
                #     cv2.imshow('result1', out)
                #     # video_writer.write(out)
                #     outsignal[comnum1][0] = 2
                #     outsignal[comnum1][1] = -1
                #     print(str(comnum1+1)+'号相机画面异常')
                #     continue
                if k1 == 0:
                    redx3 = index3[index3.argmax(axis=0)[1]][1]
                    redx3_s = index3[index3.argmin(axis=0)[1]][1]
                    record_red_length3 = redx3 - redx3_s
                    record_red_max3 = redx3_s
                elif len(index3) == 0:
                    redx3 = record_red_max3 + record_red_length3
                    redx3_s = record_red_max3
                else:
                    redx3 = index3[index3.argmax(axis=0)[1]][1]
                    redx3_s = index3[index3.argmin(axis=0)[1]][1]
                index6 = coords_green[np.where(coords_green[:, 0] == greeny3)]
                greenx3 = index6[index6.argmin(axis=0)[1]][1]

                index4 = coords_red[np.where(coords_red[:, 0] == greeny4)]
                # if len(index4) == 0 and ju == False:
                #     img = cv2.putText(img, "The object is blocked", (600, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                #                       (0, 0, 255), 2)
                #     out = cv2.addWeighted(resized_image, 0.5, img, 0.5, 0)
                #     cv2.imshow('result1', out)
                #     # video_writer.write(out)
                #     outsignal[comnum1][0] = 2
                #     outsignal[comnum1][1] = -1
                #     print(str(comnum1+1)+'号相机画面异常')
                #     continue
                if k1 == 0:
                    redx4 = index4[index4.argmax(axis=0)[1]][1]
                    redx4_s = index4[index4.argmin(axis=0)[1]][1]
                    record_red_length4 = redx4 - redx4_s
                    record_red_max4 = redx4_s
                elif len(index4) == 0:
                    redx4 = record_red_max4 + record_red_length4
                    redx4_s = record_red_max4
                else:
                    redx4 = index4[index4.argmax(axis=0)[1]][1]
                    redx4_s = index4[index4.argmin(axis=0)[1]][1]
                index7 = coords_green[np.where(coords_green[:, 0] == greeny4)]
                greenx4 = index7[index7.argmin(axis=0)[1]][1]

                # if greenx4 - greenx1 > 100:
                #     img = cv2.putText(img, "security: safe", (600, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (250, 0, 0), 2)
                #     img = cv2.putText(img, "distance: far away", (600, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (250, 0, 0), 2)
                #     # exit(0)    #image
                #     out = cv2.addWeighted(resized_image, 0.5, img, 0.5, 0)
                #     #cv2.imshow('result', out)
                #     outsignal[comnum][0]=0
                #     outsignal[comnum][1]=-1

                # video_writer.write(out)
                # print('#############################')
                # continue  # video

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
                        for x in range(min(record_red_max1 + record_red_length1, 1024), greenx1):
                            img[x, j] = (250, 0, 0)
                    for j in range(greeny2 - 3, greeny2 + 3):
                        for x in range(min(record_red_max2 + record_red_length2, 1024), greenx2):
                            img[x, j] = (250, 0, 0)
                    for j in range(greeny3 - 3, greeny3 + 3):
                        for x in range(min(record_red_max3 + record_red_length3, 1024), greenx3):
                            img[x, j] = (250, 0, 0)
                    for j in range(greeny4 - 3, greeny4 + 3):
                        for x in range(min(record_red_max4 + record_red_length4, 1024), greenx4):
                            img[x, j] = (250, 0, 0)
                else:
                    img = cv2.putText(img, "security: safe", (600, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (250, 0, 0),
                                      2)
                    img = cv2.putText(img, "distance: far away", (600, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                                      (250, 0, 0), 2)
                    # exit(0)  #image
                    out = cv2.addWeighted(resized_image, 0.5, img, 0.5, 0)
                    cv2.imshow('result1', out)
                    # video_writer.write(out)
                    outsignal[comnum1][0] = 0
                    if mindistance1<0:
                        outsignal[comnum1][1] =-1
                        print(str(comnum1+1)+'号相机画面正常,   距离:  >500')
                    else:
                        outsignal[comnum1][1] = mindistance1
                        print(str(comnum1+1)+'号相机画面正常,   距离:  '+str(mindistance1))
                    continue  # video

                mindistance1 = min(greenx1 - record_red_max1 - record_red_length1,
                                  greenx2 - record_red_max2 - record_red_length2,
                                  greenx3 - record_red_max3 - record_red_length3,
                                  greenx4 - record_red_max4 - record_red_length4)
                # mindistance=min(greenx1-record_red_max1-record_red_length1,greenx2-record_red_max2-record_red_length2,greenx3-record_red_max3-record_red_length3)

                img = cv2.putText(img, "distance: " + str(mindistance1), (600, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                                  (250, 0, 0), 2)

                # if record_red_max-redx1_s>20:
                #   img = cv2.putText(image, "security: unsafe", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (250, 0, 0), 2)
                

                if mindistance1 > int(dissss[0]):
                    img = cv2.putText(img, "security: safe", (600, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (250, 0, 0),
                                      2)
                    outsignal[comnum1][0] = 0
                    outsignal[comnum1][1] = mindistance1
                    print(str(comnum1+1)+'号相机画面正常,   距离:  '+str(mindistance1))
                else:
                    img = cv2.putText(img, "security: stop rising", (600, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                                      (0, 255, 255), 2)
                    outsignal[comnum1][0] = 1
                    if mindistance1<0:
                        outsignal[comnum1][1] = 0
                        print(str(comnum1+1)+'号相机画面正常,   距离:  0  距离较近')
                    else:
                        outsignal[comnum1][1] = mindistance1
                        print(str(comnum1+1)+'号相机画面正常,   距离:  '+str(mindistance1)+'  距离较近')
                out = cv2.addWeighted(resized_image, 0.5, img, 0.5, 0)

                cv2.imshow('result1', out)
        except:
            try:
                img = cv2.putText(img, "security: safe", (600, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (250, 0, 0),
                                          2)
                img = cv2.putText(img, "distance: far away", (600, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                                          (250, 0, 0), 2)

                out = cv2.addWeighted(resized_image, 0.5, img, 0.5, 0)
                cv2.imshow('result1', out)
                continue
            except:
                continue
            continue

def ca2():
    global outsignal
    global urlgl
    global frame_global2
    global comnum2
    global k2
    global dissss
    global jured2
    global mindistance2
    while True:
        try:
            if frame_global2.any():
                if cv2.waitKey(100) & 0xff == ord('q'):
                    break
                
                loaded_image = frame_global2
                resized_image = cv2.resize(loaded_image, (1024, 1024))
                input_image = np.expand_dims(np.float32(resized_image[:1024, :1024]), axis=0) / 255.0

                st = time.clock()

                output_image = sess.run(network, feed_dict={net_input: input_image})

                # run_time = time.clock()
                # print("run_time", run_time-st)
                # Graytime = time.clock()
                output_image = np.array(output_image[0, :, :, :])
                output_image = helpers.reverse_one_hot(output_image)

                out_vis_image = helpers.colour_code_segmentation(output_image, label_values)
                findContour = time.clock()
                img = cv2.resize(cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_RGB2BGR), (1024, 1024),
                                 interpolation=cv2.INTER_AREA)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换颜色空间
                ret, thresh = cv2.threshold(gray, 5, 100, 0)
                image, contours, hier = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                # print("找轮廓", time.clock()-findContour)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                bodynum = 0
                greenno = 1
                # f = open("out.txt", "w")
                # print(0, file = f)  #py3
                # #print >> f, 0
                # f.close()
                # start_contours1 = time.clock()
                # run_time2gray = time.clock()
                # print('toGraytime',run_time2gray-Graytime)

                for i in range(len(contours)):
                    area = cv2.contourArea(contours[i])
                    cv2.drawContours(img, contours, i, (0, 0, 0), cv2.FILLED)
                    if (area > 60000 and (max(contours[i][:, 0][:, 0]) - min(contours[i][:, 0][:, 0])) > 700 and (
                            max(contours[i][:, 0][:, 1]) - min(contours[i][:, 0][:, 1])) < 300):
                        coords_red = contours[i][:, 0]
                        coords_red_area = area
                        jured2 = 1
                        img = cv2.drawContours(img, contours, i, (128, 255, 255), 3)
                        bodynum += 1

                if jured2 == 0:
                    img = cv2.putText(img, "The object is blocked", (600, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                                      (0, 0, 255), 2)
                    out = cv2.addWeighted(resized_image, 0.5, img, 0.5, 0)
                    # cv2.imshow('result2', out)
                    # video_writer.write(out)
                    outsignal[comnum2][0] = 2
                    outsignal[comnum2][1] = -1
                    print(str(comnum2+1)+'号相机画面异常')
                    # end_contours1 = time.clock()
                    # print('contours1', end_contours1 - start_contours1)
                    continue
                # start_contours2 = time.clock()
                for i in range(len(contours)):
                    area = cv2.contourArea(contours[i])
                    if (area > 60000 and max(contours[i][:, 0][:, 0]) - min(contours[i][:, 0][:, 0]) > 420 and (
                            max(contours[i][:, 0][:, 1]) - min(contours[i][:, 0][:, 1])) >= 250 and min(
                        contours[i][:, 0][:, 1]) > min(coords_red[:, 1]) + 5):
                        coords_green = contours[i][:, 0]
                        coords_green_area = area
                        greenno = 0
                        bodynum += 1
                        img = cv2.drawContours(img, contours, i, (128, 255, 255), 3)
                        # end_contours2 = time.clock()
                        # print('contours2', end_contours2 - start_contours2)
                
                if bodynum < 1  and mindistance2<int(dissss[1]) and mindistance2!=-1:
                    img = cv2.putText(img, "security: stop rising", (600, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                                      (0, 255, 255), 2)
                    outsignal[comnum2][0] = 1
                    
                    outsignal[comnum2][1] = 0
                    print(str(comnum2+1)+'号相机画面正常,   距离:  0  距离较近')
                    
                    out = cv2.addWeighted(resized_image, 0.5, img, 0.5, 0)
                    continue

                    


                greenex = 'coords_green_area' in locals().keys()

                if greenex == False or greenno == 1:
                    img = cv2.putText(img, "security: safe", (600, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (250, 0, 0),
                                      2)
                    img = cv2.putText(img, "distance: far away", (600, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                                      (250, 0, 0), 2)

                    out = cv2.addWeighted(resized_image, 0.5, img, 0.5, 0)
                    # cv2.imshow('result2', out)
                    outsignal[comnum2][0] = 0
                    
                    if mindistance2<0:
                        outsignal[comnum2][1] =-1
                        print(str(comnum2+1)+'号相机画面正常,   距离:  >500')
                    else:
                        outsignal[comnum2][1] = mindistance2
                        print(str(comnum2+1)+'号相机画面正常,   距离:  '+str(mindistance2))
                    continue  # video

                [greenminx_index, greenminy_index] = coords_green.argmin(axis=0)
                

                greeny1 = coords_green[greenminy_index][0]
                if greeny1<int(dissss[2]) or greeny1>int(dissss[3]):
                    greeny1 = 470

                
                greeny2 = 420
                greeny3 = 580

                greeny4 = 530

                green_length_threshold_max = 700
                green_length_threshold = greeny3 - greeny2

                # if green_length_threshold < 420:
                #     img = cv2.putText(img, "security: safe", (600,100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (250, 0, 0), 2)
                #     img = cv2.putText(img, "distance: far away", (600, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (250, 0, 0), 2)
                #     # exit(0)    #imag
                #     out = cv2.addWeighted(resized_image, 0.5, img, 0.5, 0)
                    #cv2.imshow('result2', out)
                #     # video_writer.write(out)
                #     outsignal[comnum][0]=0
                #     outsignal[comnum][1]=-1

                #     continue  # video

                # red lower
                index = coords_red[np.where(coords_red[:, 0] == greeny1)]

                ju = 'record_red_max1' in locals().keys()
                k2 += 1
                # if len(index) == 0 and ju == False:
                #     img = cv2.putText(img, "The object is blocked", (600, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                #                       (0, 0, 255), 2)
                #     out = cv2.addWeighted(resized_image, 0.5, img, 0.5, 0)
                #     # cv2.imshow('result2', out)
                #     # video_writer.write(out)
                #     outsignal[comnum2][0] = 2
                #     outsignal[comnum2][1] = -1
                #     k2-=1
                #     print(str(comnum2+1)+'号相机画面异常')
                #     continue
                if k2 == 0:  # fixed the measurement bug when changinging videos
                    redx1 = index[index.argmax(axis=0)[1]][1]
                    redx1_s = index[index.argmin(axis=0)[1]][1]
                    record_red_length1 = redx1 - redx1_s
                    record_red_max1 = redx1_s
                elif len(index) == 0:
                    redx1 = record_red_max1 + record_red_length1
                    redx1_s = record_red_max1
                else:
                    redx1 = index[index.argmax(axis=0)[1]][1]
                    redx1_s = index[index.argmin(axis=0)[1]][1]
                index2 = coords_red[np.where(coords_red[:, 0] == greeny2)]

                index10=coords_green[np.where(coords_green[:,0]==greeny1)]
                greenx1=index10[index10.argmin(axis=0)[1]][1]
                # if len(index2) == 0 and ju == False:
                #     img = cv2.putText(img, "The object is blocked", (600, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                #                       (0, 0, 255), 2)
                #     out = cv2.addWeighted(resized_image, 0.5, img, 0.5, 0)
                #     # cv2.imshow('result2', out)
                #     # video_writer.write(out)
                #     outsignal[comnum2][0] = 2
                #     outsignal[comnum2][1] = -1
                #     print(str(comnum2+1)+'号相机画面异常')
                #     continue
                if k2 == 0:
                    redx2 = index2[index2.argmax(axis=0)[1]][1]
                    redx2_s = index2[index2.argmin(axis=0)[1]][1]
                    record_red_length2 = redx2 - redx2_s
                    record_red_max2 = redx2_s
                elif len(index2) == 0:
                    redx2 = record_red_max2 + record_red_length2
                    redx2_s = record_red_max2
                else:
                    redx2 = index2[index2.argmax(axis=0)[1]][1]
                    redx2_s = index2[index2.argmin(axis=0)[1]][1]
                index5 = coords_green[np.where(coords_green[:, 0] == greeny2)]
                greenx2 = index5[index5.argmin(axis=0)[1]][1]

                index3 = coords_red[np.where(coords_red[:, 0] == greeny3)]
                # if len(index3) == 0 and ju == False:
                #     img = cv2.putText(img, "The object is blocked", (600, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                #                       (0, 0, 255), 2)
                #     out = cv2.addWeighted(resized_image, 0.5, img, 0.5, 0)
                #     # cv2.imshow('result2', out)
                #     # video_writer.write(out)
                #     outsignal[comnum2][0] = 2
                #     outsignal[comnum2][1] = -1
                #     print(str(comnum2+1)+'号相机画面异常')
                #     continue
                if k2 == 0:
                    redx3 = index3[index3.argmax(axis=0)[1]][1]
                    redx3_s = index3[index3.argmin(axis=0)[1]][1]
                    record_red_length3 = redx3 - redx3_s
                    record_red_max3 = redx3_s
                elif len(index3) == 0:
                    redx3 = record_red_max3 + record_red_length3
                    redx3_s = record_red_max3
                else:
                    redx3 = index3[index3.argmax(axis=0)[1]][1]
                    redx3_s = index3[index3.argmin(axis=0)[1]][1]
                index6 = coords_green[np.where(coords_green[:, 0] == greeny3)]
                greenx3 = index6[index6.argmin(axis=0)[1]][1]

                index4 = coords_red[np.where(coords_red[:, 0] == greeny4)]
                # if len(index4) == 0 and ju == False:
                #     img = cv2.putText(img, "The object is blocked", (600, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                #                       (0, 0, 255), 2)
                #     out = cv2.addWeighted(resized_image, 0.5, img, 0.5, 0)
                #     # cv2.imshow('result2', out)
                #     # video_writer.write(out)
                #     outsignal[comnum2][0] = 2
                #     outsignal[comnum2][1] = -1
                #     print(str(comnum2+1)+'号相机画面异常')
                #     continue
                if k2 == 0:
                    redx4 = index4[index4.argmax(axis=0)[1]][1]
                    redx4_s = index4[index4.argmin(axis=0)[1]][1]
                    record_red_length4 = redx4 - redx4_s
                    record_red_max4 = redx4_s
                elif len(index4) == 0:
                    redx4 = record_red_max4 + record_red_length4
                    redx4_s = record_red_max4
                else:
                    redx4 = index4[index4.argmax(axis=0)[1]][1]
                    redx4_s = index4[index4.argmin(axis=0)[1]][1]
                index7 = coords_green[np.where(coords_green[:, 0] == greeny4)]
                greenx4 = index7[index7.argmin(axis=0)[1]][1]

                # if greenx4 - greenx1 > 100:
                #     img = cv2.putText(img, "security: safe", (600, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (250, 0, 0), 2)
                #     img = cv2.putText(img, "distance: far away", (600, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (250, 0, 0), 2)
                #     # exit(0)    #image
                #     out = cv2.addWeighted(resized_image, 0.5, img, 0.5, 0)
                #     #cv2.imshow('result', out)
                #     outsignal[comnum][0]=0
                #     outsignal[comnum][1]=-1

                # video_writer.write(out)
                # print('#############################')
                # continue  # video

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
                        for x in range(min(record_red_max1 + record_red_length1, 1024), greenx1):
                            img[x, j] = (250, 0, 0)
                    for j in range(greeny2 - 3, greeny2 + 3):
                        for x in range(min(record_red_max2 + record_red_length2, 1024), greenx2):
                            img[x, j] = (250, 0, 0)
                    for j in range(greeny3 - 3, greeny3 + 3):
                        for x in range(min(record_red_max3 + record_red_length3, 1024), greenx3):
                            img[x, j] = (250, 0, 0)
                    for j in range(greeny4 - 3, greeny4 + 3):
                        for x in range(min(record_red_max4 + record_red_length4, 1024), greenx4):
                            img[x, j] = (250, 0, 0)
                else:
                    img = cv2.putText(img, "security: safe", (600, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (250, 0, 0),
                                      2)
                    img = cv2.putText(img, "distance: far away", (600, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                                      (250, 0, 0), 2)
                    # exit(0)  #image
                    out = cv2.addWeighted(resized_image, 0.5, img, 0.5, 0)
                    # cv2.imshow('result2', out)
                    # video_writer.write(out)
                    outsignal[comnum2][0] = 0
                    if mindistance2<0:
                        outsignal[comnum2][1] =-1
                        print(str(comnum2+1)+'号相机画面正常,   距离:  >500')
                    else:
                        outsignal[comnum2][1] = mindistance2
                        print(str(comnum2+1)+'号相机画面正常,   距离:  '+str(mindistance2))
                    continue  # video

                mindistance2 = min(greenx1 - record_red_max1 - record_red_length1,
                                  greenx2 - record_red_max2 - record_red_length2,
                                  greenx3 - record_red_max3 - record_red_length3,
                                  greenx4 - record_red_max4 - record_red_length4)
                # mindistance=min(greenx1-record_red_max1-record_red_length1,greenx2-record_red_max2-record_red_length2,greenx3-record_red_max3-record_red_length3)

                img = cv2.putText(img, "distance: " + str(mindistance2), (600, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                                  (250, 0, 0), 2)

                # if record_red_max-redx1_s>20:
                #   img = cv2.putText(image, "security: unsafe", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (250, 0, 0), 2)
                

                if mindistance2 > int(dissss[0]):
                    img = cv2.putText(img, "security: safe", (600, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (250, 0, 0),
                                      2)
                    outsignal[comnum2][0] = 0
                    outsignal[comnum2][1] = mindistance2
                    print(str(comnum2+1)+'号相机画面正常,   距离:  '+str(mindistance2))
                else:
                    img = cv2.putText(img, "security: stop rising", (600, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                                      (0, 255, 255), 2)
                    outsignal[comnum2][0] = 1
                    if mindistance2<0:
                        outsignal[comnum2][1] = 0
                        print(str(comnum2+1)+'号相机画面正常,   距离:  0  距离较近')
                    else:
                        outsignal[comnum2][1] = mindistance2
                        print(str(comnum2+1)+'号相机画面正常,   距离:  '+str(mindistance2)+'  距离较近')
                out = cv2.addWeighted(resized_image, 0.5, img, 0.5, 0)

                # cv2.imshow('result2', out)

        except:
            continue

def ca3():
    global outsignal
    global urlgl
    global frame_global3
    global comnum3
    global k3
    global dissss
    global jured3
    global mindistance3
    while True:
        try:
            if frame_global3.any():
                if cv2.waitKey(100) & 0xff == ord('q'):
                    break
                
                loaded_image = frame_global3
                resized_image = cv2.resize(loaded_image, (1024, 1024))
                input_image = np.expand_dims(np.float32(resized_image[:1024, :1024]), axis=0) / 255.0

                st = time.clock()

                output_image = sess.run(network, feed_dict={net_input: input_image})

                # run_time = time.clock()
                # print("run_time", run_time-st)
                # Graytime = time.clock()
                output_image = np.array(output_image[0, :, :, :])
                output_image = helpers.reverse_one_hot(output_image)

                out_vis_image = helpers.colour_code_segmentation(output_image, label_values)
                # findContour = time.clock()
                img = cv2.resize(cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_RGB2BGR), (1024, 1024),
                                 interpolation=cv2.INTER_AREA)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换颜色空间
                ret, thresh = cv2.threshold(gray, 5, 100, 0)
                image, contours, hier = cv2.findContours (thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                # print("找轮廓", time.clock()-findContour)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                bodynum = 0
                greenno = 1
                # f = open("out.txt", "w")
                # print(0, file = f)  #py3
                # #print >> f, 0
                # f.close()
                # start_contours1 = time.clock()
                # run_time2gray = time.clock()
                # print('toGraytime',run_time2gray-Graytime)

                for i in range(len(contours)):
                    area = cv2.contourArea(contours[i])
                    cv2.drawContours(img, contours, i, (0, 0, 0), cv2.FILLED)
                    if (area > 60000 and (max(contours[i][:, 0][:, 0]) - min(contours[i][:, 0][:, 0])) > 700 and (
                            max(contours[i][:, 0][:, 1]) - min(contours[i][:, 0][:, 1])) < 300):
                        coords_red = contours[i][:, 0]
                        coords_red_area = area
                        jured3 = 1
                        img = cv2.drawContours(img, contours, i, (128, 255, 255), 3)
                        bodynum += 1

                if jured3 == 0:
                    img = cv2.putText(img, "The object is blocked", (600, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                                      (0, 0, 255), 2)
                    out = cv2.addWeighted(resized_image, 0.5, img, 0.5, 0)
                    # cv2.imshow('result', out)
                    # video_writer.write(out)
                    outsignal[comnum3][0] = 2
                    outsignal[comnum3][1] = -1
                    print(str(comnum3+1)+'号相机画面异常')
                    # end_contours1 = time.clock()
                    # print('contours1', end_contours1 - start_contours1)
                    continue
                # start_contours2 = time.clock()
                for i in range(len(contours)):
                    area = cv2.contourArea(contours[i])
                    if (area > 60000 and max(contours[i][:, 0][:, 0]) - min(contours[i][:, 0][:, 0]) > 420 and (
                            max(contours[i][:, 0][:, 1]) - min(contours[i][:, 0][:, 1])) >= 250 and min(
                        contours[i][:, 0][:, 1]) > min(coords_red[:, 1]) + 5):
                        coords_green = contours[i][:, 0]
                        coords_green_area = area
                        greenno = 0
                        bodynum += 1
                        img = cv2.drawContours(img, contours, i, (128, 255, 255), 3)
                        # end_contours2 = time.clock()
                        # print('contours2', end_contours2 - start_contours2)
                

                if bodynum < 1 and mindistance3<int(dissss[1]) and mindistance3!=-1:
                    img = cv2.putText(img, "security: stop rising", (600, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                                      (0, 255, 255), 2)
                    outsignal[comnum3][0] = 1
                    
                    outsignal[comnum3][1] = 0
                    print(str(comnum3+1)+'号相机画面正常,   距离:  0  距离较近')
                    
                    out = cv2.addWeighted(resized_image, 0.5, img, 0.5, 0)
                    continue

                    


                greenex = 'coords_green_area' in locals().keys()

                if greenex == False or greenno == 1:
                    img = cv2.putText(img, "security: safe", (600, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (250, 0, 0),
                                      2)
                    img = cv2.putText(img, "distance: far away", (600, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                                      (250, 0, 0), 2)

                    out = cv2.addWeighted(resized_image, 0.5, img, 0.5, 0)
                    # cv2.imshow('result', out)
                    outsignal[comnum3][0] = 0
                    
                    if mindistance3<0:
                        outsignal[comnum3][1] =-1
                        print(str(comnum3+1)+'号相机画面正常,   距离:  >500')
                    else:
                        outsignal[comnum3][1] = mindistance3
                        print(str(comnum3+1)+'号相机画面正常,   距离:  '+str(mindistance3))
                    continue  # video

                [greenminx_index, greenminy_index] = coords_green.argmin(axis=0)
                

                greeny1 = coords_green[greenminy_index][0]
                if greeny1<int(dissss[2]) or greeny1>int(dissss[3]):
                    greeny1 = 470

                
                greeny2 = 420
                greeny3 = 580

                greeny4 = 530

                green_length_threshold_max = 700
                green_length_threshold = greeny3 - greeny2

                # if green_length_threshold < 420:
                #     img = cv2.putText(img, "security: safe", (600,100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (250, 0, 0), 2)
                #     img = cv2.putText(img, "distance: far away", (600, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (250, 0, 0), 2)
                #     # exit(0)    #imag
                #     out = cv2.addWeighted(resized_image, 0.5, img, 0.5, 0)
                #     #cv2.imshow('result', out)
                #     # video_writer.write(out)
                #     outsignal[comnum][0]=0
                #     outsignal[comnum][1]=-1

                #     continue  # video

                # red lower
                index = coords_red[np.where(coords_red[:, 0] == greeny1)]

                ju = 'record_red_max1' in locals().keys()
                k3 += 1
                # if len(index) == 0 and ju == False:
                #     img = cv2.putText(img, "The object is blocked", (600, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                #                       (0, 0, 255), 2)
                #     out = cv2.addWeighted(resized_image, 0.5, img, 0.5, 0)
                #     # cv2.imshow('result', out)
                #     # video_writer.write(out)
                #     outsignal[comnum3][0] = 2
                #     outsignal[comnum3][1] = -1
                #     k3-=1
                #     print(str(comnum3+1)+'号相机画面异常')
                #     continue
                if k3 == 0:  # fixed the measurement bug when changinging videos
                    redx1 = index[index.argmax(axis=0)[1]][1]
                    redx1_s = index[index.argmin(axis=0)[1]][1]
                    record_red_length1 = redx1 - redx1_s
                    record_red_max1 = redx1_s
                elif len(index) == 0:
                    redx1 = record_red_max1 + record_red_length1
                    redx1_s = record_red_max1
                else:
                    redx1 = index[index.argmax(axis=0)[1]][1]
                    redx1_s = index[index.argmin(axis=0)[1]][1]
                index2 = coords_red[np.where(coords_red[:, 0] == greeny2)]

                index10=coords_green[np.where(coords_green[:,0]==greeny1)]
                greenx1=index10[index10.argmin(axis=0)[1]][1]
                # if len(index2) == 0 and ju == False:
                #     img = cv2.putText(img, "The object is blocked", (600, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                #                       (0, 0, 255), 2)
                #     out = cv2.addWeighted(resized_image, 0.5, img, 0.5, 0)
                #     # cv2.imshow('result', out)
                #     # video_writer.write(out)
                #     outsignal[comnum3][0] = 2
                #     outsignal[comnum3][1] = -1
                #     print(str(comnum3+1)+'号相机画面异常')
                #     continue
                if k3 == 0:
                    redx2 = index2[index2.argmax(axis=0)[1]][1]
                    redx2_s = index2[index2.argmin(axis=0)[1]][1]
                    record_red_length2 = redx2 - redx2_s
                    record_red_max2 = redx2_s
                elif len(index2) == 0:
                    redx2 = record_red_max2 + record_red_length2
                    redx2_s = record_red_max2
                else:
                    redx2 = index2[index2.argmax(axis=0)[1]][1]
                    redx2_s = index2[index2.argmin(axis=0)[1]][1]
                index5 = coords_green[np.where(coords_green[:, 0] == greeny2)]
                greenx2 = index5[index5.argmin(axis=0)[1]][1]

                index3 = coords_red[np.where(coords_red[:, 0] == greeny3)]
                # if len(index3) == 0 and ju == False:
                #     img = cv2.putText(img, "The object is blocked", (600, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                #                       (0, 0, 255), 2)
                #     out = cv2.addWeighted(resized_image, 0.5, img, 0.5, 0)
                #     # cv2.imshow('result', out)
                #     # video_writer.write(out)
                #     outsignal[comnum3][0] = 2
                #     outsignal[comnum3][1] = -1
                #     print(str(comnum3+1)+'号相机画面异常')
                #     continue
                if k3 == 0:
                    redx3 = index3[index3.argmax(axis=0)[1]][1]
                    redx3_s = index3[index3.argmin(axis=0)[1]][1]
                    record_red_length3 = redx3 - redx3_s
                    record_red_max3 = redx3_s
                elif len(index3) == 0:
                    redx3 = record_red_max3 + record_red_length3
                    redx3_s = record_red_max3
                else:
                    redx3 = index3[index3.argmax(axis=0)[1]][1]
                    redx3_s = index3[index3.argmin(axis=0)[1]][1]
                index6 = coords_green[np.where(coords_green[:, 0] == greeny3)]
                greenx3 = index6[index6.argmin(axis=0)[1]][1]

                index4 = coords_red[np.where(coords_red[:, 0] == greeny4)]
                # if len(index4) == 0 and ju == False:
                #     img = cv2.putText(img, "The object is blocked", (600, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                #                       (0, 0, 255), 2)
                #     out = cv2.addWeighted(resized_image, 0.5, img, 0.5, 0)
                #     # cv2.imshow('result', out)
                #     # video_writer.write(out)
                #     outsignal[comnum3][0] = 2
                #     outsignal[comnum3][1] = -1
                #     print(str(comnum3+1)+'号相机画面异常')
                #     continue
                if k3 == 0:
                    redx4 = index4[index4.argmax(axis=0)[1]][1]
                    redx4_s = index4[index4.argmin(axis=0)[1]][1]
                    record_red_length4 = redx4 - redx4_s
                    record_red_max4 = redx4_s
                elif len(index4) == 0:
                    redx4 = record_red_max4 + record_red_length4
                    redx4_s = record_red_max4
                else:
                    redx4 = index4[index4.argmax(axis=0)[1]][1]
                    redx4_s = index4[index4.argmin(axis=0)[1]][1]
                index7 = coords_green[np.where(coords_green[:, 0] == greeny4)]
                greenx4 = index7[index7.argmin(axis=0)[1]][1]

                # if greenx4 - greenx1 > 100:
                #     img = cv2.putText(img, "security: safe", (600, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (250, 0, 0), 2)
                #     img = cv2.putText(img, "distance: far away", (600, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (250, 0, 0), 2)
                #     # exit(0)    #image
                #     out = cv2.addWeighted(resized_image, 0.5, img, 0.5, 0)
                #     #cv2.imshow('result', out)
                #     outsignal[comnum][0]=0
                #     outsignal[comnum][1]=-1

                # video_writer.write(out)
                # print('#############################')
                # continue  # video

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
                        for x in range(min(record_red_max1 + record_red_length1, 1024), greenx1):
                            img[x, j] = (250, 0, 0)
                    for j in range(greeny2 - 3, greeny2 + 3):
                        for x in range(min(record_red_max2 + record_red_length2, 1024), greenx2):
                            img[x, j] = (250, 0, 0)
                    for j in range(greeny3 - 3, greeny3 + 3):
                        for x in range(min(record_red_max3 + record_red_length3, 1024), greenx3):
                            img[x, j] = (250, 0, 0)
                    for j in range(greeny4 - 3, greeny4 + 3):
                        for x in range(min(record_red_max4 + record_red_length4, 1024), greenx4):
                            img[x, j] = (250, 0, 0)
                else:
                    img = cv2.putText(img, "security: safe", (600, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (250, 0, 0),
                                      2)
                    img = cv2.putText(img, "distance: far away", (600, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                                      (250, 0, 0), 2)
                    # exit(0)  #image
                    out = cv2.addWeighted(resized_image, 0.5, img, 0.5, 0)
                    # cv2.imshow('result', out)
                    # video_writer.write(out)
                    outsignal[comnum3][0] = 0
                    if mindistance3<0:
                        outsignal[comnum3][1] =-1
                        print(str(comnum3+1)+'号相机画面正常,   距离:  >500')
                    else:
                        outsignal[comnum3][1] = mindistance3
                        print(str(comnum3+1)+'号相机画面正常,   距离:  '+str(mindistance3))
                    continue  # video

                mindistance3 = min(greenx1 - record_red_max1 - record_red_length1,
                                  greenx2 - record_red_max2 - record_red_length2,
                                  greenx3 - record_red_max3 - record_red_length3,
                                  greenx4 - record_red_max4 - record_red_length4)
                # mindistance=min(greenx1-record_red_max1-record_red_length1,greenx2-record_red_max2-record_red_length2,greenx3-record_red_max3-record_red_length3)

                img = cv2.putText(img, "distance: " + str(mindistance3), (600, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                                  (250, 0, 0), 2)

                # if record_red_max-redx1_s>20:
                #   img = cv2.putText(image, "security: unsafe", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (250, 0, 0), 2)
                

                if mindistance3 > int(dissss[0]):
                    img = cv2.putText(img, "security: safe", (600, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (250, 0, 0),
                                      2)
                    outsignal[comnum3][0] = 0
                    outsignal[comnum3][1] = mindistance3
                    print(str(comnum3+1)+'号相机画面正常,   距离:  '+str(mindistance3))
                else:
                    img = cv2.putText(img, "security: stop rising", (600, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                                      (0, 255, 255), 2)
                    outsignal[comnum3][0] = 1
                    if mindistance3<0:
                        outsignal[comnum3][1] = 0
                        print(str(comnum3+1)+'号相机画面正常,   距离:  0  距离较近')
                    else:
                        outsignal[comnum3][1] = mindistance3
                        print(str(comnum3+1)+'号相机画面正常,   距离:  '+str(mindistance3)+'  距离较近')
                out = cv2.addWeighted(resized_image, 0.5, img, 0.5, 0)

                # cv2.imshow('result', out)

        except:
            continue
def ca4():
    global outsignal
    global urlgl
    global frame_global4
    global comnum4
    global k4
    global jured4
    global dissss
    global mindistance4
    while True:
        try:
            if frame_global4.any():
                if cv2.waitKey(100) & 0xff == ord('q'):
                    break
                
                loaded_image = frame_global4
                resized_image = cv2.resize(loaded_image, (1024, 1024))
                input_image = np.expand_dims(np.float32(resized_image[:1024, :1024]), axis=0) / 255.0

                st = time.clock()

                output_image = sess.run(network, feed_dict={net_input: input_image})

                # run_time = time.clock()
                # print("run_time", run_time-st)
                # Graytime = time.clock()
                output_image = np.array(output_image[0, :, :, :])
                output_image = helpers.reverse_one_hot(output_image)

                out_vis_image = helpers.colour_code_segmentation(output_image, label_values)
                findContour = time.clock()
                img = cv2.resize(cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_RGB2BGR), (1024, 1024),
                                 interpolation=cv2.INTER_AREA)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换颜色空间
                ret, thresh = cv2.threshold(gray, 5, 100, 0)
                image, contours, hier = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                # print("找轮廓", time.clock()-findContour)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                bodynum = 0
                greenno = 1
                # f = open("out.txt", "w")
                # print(0, file = f)  #py3
                # #print >> f, 0
                # f.close()
                # start_contours1 = time.clock()
                # run_time2gray = time.clock()
                # print('toGraytime',run_time2gray-Graytime)

                for i in range(len(contours)):
                    area = cv2.contourArea(contours[i])
                    cv2.drawContours(img, contours, i, (0, 0, 0), cv2.FILLED)
                    if (area > 60000 and (max(contours[i][:, 0][:, 0]) - min(contours[i][:, 0][:, 0])) > 700 and (
                            max(contours[i][:, 0][:, 1]) - min(contours[i][:, 0][:, 1])) < 300):
                        coords_red = contours[i][:, 0]
                        coords_red_area = area
                        jured4 = 1
                        img = cv2.drawContours(img, contours, i, (128, 255, 255), 3)
                        bodynum += 1

                if jured4 == 0:
                    img = cv2.putText(img, "The object is blocked", (600, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                                      (0, 0, 255), 2)
                    out = cv2.addWeighted(resized_image, 0.5, img, 0.5, 0)
                    # cv2.imshow('result', out)
                    # video_writer.write(out)
                    outsignal[comnum4][0] = 2
                    outsignal[comnum4][1] = -1
                    print(str(comnum4+1)+'号相机画面异常')
                    # end_contours1 = time.clock()
                    # print('contours1', end_contours1 - start_contours1)
                    continue
                # start_contours2 = time.clock()
                for i in range(len(contours)):
                    area = cv2.contourArea(contours[i])
                    if (area > 60000 and max(contours[i][:, 0][:, 0]) - min(contours[i][:, 0][:, 0]) > 420 and (
                            max(contours[i][:, 0][:, 1]) - min(contours[i][:, 0][:, 1])) >= 250 and min(
                        contours[i][:, 0][:, 1]) > min(coords_red[:, 1]) + 5):
                        coords_green = contours[i][:, 0]
                        coords_green_area = area
                        greenno = 0
                        bodynum += 1
                        img = cv2.drawContours(img, contours, i, (128, 255, 255), 3)
                        # end_contours2 = time.clock()
                        # print('contours2', end_contours2 - start_contours2)
                


                if bodynum < 1 and mindistance4<int(dissss[1]) and mindistance4!=-1:
                    img = cv2.putText(img, "security: stop rising", (600, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                                      (0, 255, 255), 2)
                    outsignal[comnum4][0] = 1
                    
                    outsignal[comnum4][1] = 0
                    print(str(comnum4+1)+'号相机画面正常,   距离:  0  距离较近')
                    
                    out = cv2.addWeighted(resized_image, 0.5, img, 0.5, 0)
                    continue

                    




                greenex = 'coords_green_area' in locals().keys()

                if greenex == False or greenno == 1:
                    img = cv2.putText(img, "security: safe", (600, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (250, 0, 0),
                                      2)
                    img = cv2.putText(img, "distance: far away", (600, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                                      (250, 0, 0), 2)

                    out = cv2.addWeighted(resized_image, 0.5, img, 0.5, 0)
                    # cv2.imshow('result', out)
                    outsignal[comnum4][0] = 0
                    
                    if mindistance4<0:
                        outsignal[comnum4][1] =-1
                        print(str(comnum4+1)+'号相机画面正常,   距离:  >500')
                    else:
                        outsignal[comnum4][1] = mindistance4
                        print(str(comnum4+1)+'号相机画面正常,   距离:  '+str(mindistance4))
                    continue  # video

                [greenminx_index, greenminy_index] = coords_green.argmin(axis=0)
                

                greeny1 = coords_green[greenminy_index][0]
                if greeny1<int(dissss[2]) or greeny1>int(dissss[3]):
                    greeny1 = 470

                
                greeny2 = 420
                greeny3 = 580

                greeny4 = 530

                green_length_threshold_max = 700
                green_length_threshold = greeny3 - greeny2

                # if green_length_threshold < 420:
                #     img = cv2.putText(img, "security: safe", (600,100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (250, 0, 0), 2)
                #     img = cv2.putText(img, "distance: far away", (600, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (250, 0, 0), 2)
                #     # exit(0)    #imag
                #     out = cv2.addWeighted(resized_image, 0.5, img, 0.5, 0)
                #     #cv2.imshow('result', out)
                #     # video_writer.write(out)
                #     outsignal[comnum][0]=0
                #     outsignal[comnum][1]=-1

                #     continue  # video

                # red lower
                index = coords_red[np.where(coords_red[:, 0] == greeny1)]

                ju = 'record_red_max1' in locals().keys()
                k4 += 1
                # if len(index) == 0 and ju == False:
                #     img = cv2.putText(img, "The object is blocked", (600, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                #                       (0, 0, 255), 2)
                #     out = cv2.addWeighted(resized_image, 0.5, img, 0.5, 0)
                #     # cv2.imshow('result', out)
                #     # video_writer.write(out)
                #     outsignal[comnum4][0] = 2
                #     outsignal[comnum4][1] = -1
                #     k4-=1
                #     print(str(comnum4+1)+'号相机画面异常')
                #     continue
                if k4 == 0:  # fixed the measurement bug when changinging videos
                    redx1 = index[index.argmax(axis=0)[1]][1]
                    redx1_s = index[index.argmin(axis=0)[1]][1]
                    record_red_length1 = redx1 - redx1_s
                    record_red_max1 = redx1_s
                elif len(index) == 0:
                    redx1 = record_red_max1 + record_red_length1
                    redx1_s = record_red_max1
                else:
                    redx1 = index[index.argmax(axis=0)[1]][1]
                    redx1_s = index[index.argmin(axis=0)[1]][1]
                index2 = coords_red[np.where(coords_red[:, 0] == greeny2)]

                index10=coords_green[np.where(coords_green[:,0]==greeny1)]
                greenx1=index10[index10.argmin(axis=0)[1]][1]

                # if len(index2) == 0 and ju == False:
                #     img = cv2.putText(img, "The object is blocked", (600, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                #                       (0, 0, 255), 2)
                #     out = cv2.addWeighted(resized_image, 0.5, img, 0.5, 0)
                #     # cv2.imshow('result', out)
                #     # video_writer.write(out)
                #     outsignal[comnum4][0] = 2
                #     outsignal[comnum4][1] = -1
                #     print(str(comnum4+1)+'号相机画面异常')
                #     continue
                if k4 == 0:
                    redx2 = index2[index2.argmax(axis=0)[1]][1]
                    redx2_s = index2[index2.argmin(axis=0)[1]][1]
                    record_red_length2 = redx2 - redx2_s
                    record_red_max2 = redx2_s
                elif len(index2) == 0:
                    redx2 = record_red_max2 + record_red_length2
                    redx2_s = record_red_max2
                else:
                    redx2 = index2[index2.argmax(axis=0)[1]][1]
                    redx2_s = index2[index2.argmin(axis=0)[1]][1]
                index5 = coords_green[np.where(coords_green[:, 0] == greeny2)]
                greenx2 = index5[index5.argmin(axis=0)[1]][1]

                index3 = coords_red[np.where(coords_red[:, 0] == greeny3)]
                # if len(index3) == 0 and ju == False:
                #     img = cv2.putText(img, "The object is blocked", (600, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                #                       (0, 0, 255), 2)
                #     out = cv2.addWeighted(resized_image, 0.5, img, 0.5, 0)
                #     # cv2.imshow('result', out)
                #     # video_writer.write(out)
                #     outsignal[comnum4][0] = 2
                #     outsignal[comnum4][1] = -1
                #     print(str(comnum4+1)+'号相机画面异常')
                #     continue
                if k4 == 0:
                    redx3 = index3[index3.argmax(axis=0)[1]][1]
                    redx3_s = index3[index3.argmin(axis=0)[1]][1]
                    record_red_length3 = redx3 - redx3_s
                    record_red_max3 = redx3_s
                elif len(index3) == 0:
                    redx3 = record_red_max3 + record_red_length3
                    redx3_s = record_red_max3
                else:
                    redx3 = index3[index3.argmax(axis=0)[1]][1]
                    redx3_s = index3[index3.argmin(axis=0)[1]][1]
                index6 = coords_green[np.where(coords_green[:, 0] == greeny3)]
                greenx3 = index6[index6.argmin(axis=0)[1]][1]

                index4 = coords_red[np.where(coords_red[:, 0] == greeny4)]
                # if len(index4) == 0 and ju == False:
                #     img = cv2.putText(img, "The object is blocked", (600, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                #                       (0, 0, 255), 2)
                #     out = cv2.addWeighted(resized_image, 0.5, img, 0.5, 0)
                #     # cv2.imshow('result', out)
                #     # video_writer.write(out)
                #     outsignal[comnum4][0] = 2
                #     outsignal[comnum4][1] = -1
                #     print(str(comnum4+1)+'号相机画面异常')
                #     continue
                if k4 == 0:
                    redx4 = index4[index4.argmax(axis=0)[1]][1]
                    redx4_s = index4[index4.argmin(axis=0)[1]][1]
                    record_red_length4 = redx4 - redx4_s
                    record_red_max4 = redx4_s
                elif len(index4) == 0:
                    redx4 = record_red_max4 + record_red_length4
                    redx4_s = record_red_max4
                else:
                    redx4 = index4[index4.argmax(axis=0)[1]][1]
                    redx4_s = index4[index4.argmin(axis=0)[1]][1]
                index7 = coords_green[np.where(coords_green[:, 0] == greeny4)]
                greenx4 = index7[index7.argmin(axis=0)[1]][1]

                # if greenx4 - greenx1 > 100:
                #     img = cv2.putText(img, "security: safe", (600, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (250, 0, 0), 2)
                #     img = cv2.putText(img, "distance: far away", (600, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (250, 0, 0), 2)
                #     # exit(0)    #image
                #     out = cv2.addWeighted(resized_image, 0.5, img, 0.5, 0)
                #     #cv2.imshow('result', out)
                #     outsignal[comnum][0]=0
                #     outsignal[comnum][1]=-1

                # video_writer.write(out)
                # print('#############################')
                # continue  # video

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
                        for x in range(min(record_red_max1 + record_red_length1, 1024), greenx1):
                            img[x, j] = (250, 0, 0)
                    for j in range(greeny2 - 3, greeny2 + 3):
                        for x in range(min(record_red_max2 + record_red_length2, 1024), greenx2):
                            img[x, j] = (250, 0, 0)
                    for j in range(greeny3 - 3, greeny3 + 3):
                        for x in range(min(record_red_max3 + record_red_length3, 1024), greenx3):
                            img[x, j] = (250, 0, 0)
                    for j in range(greeny4 - 3, greeny4 + 3):
                        for x in range(min(record_red_max4 + record_red_length4, 1024), greenx4):
                            img[x, j] = (250, 0, 0)
                else:
                    img = cv2.putText(img, "security: safe", (600, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (250, 0, 0),
                                      2)
                    img = cv2.putText(img, "distance: far away", (600, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                                      (250, 0, 0), 2)
                    # exit(0)  #image
                    out = cv2.addWeighted(resized_image, 0.5, img, 0.5, 0)
                    # cv2.imshow('result', out)
                    # video_writer.write(out)
                    outsignal[comnum4][0] = 0
                    if mindistance4<0:
                        outsignal[comnum4][1] =-1
                        print(str(comnum4+1)+'号相机画面正常,   距离:  >500')
                    else:
                        outsignal[comnum4][1] = mindistance4
                        print(str(comnum4+1)+'号相机画面正常,   距离:  '+str(mindistance4))
                    continue  # video

                mindistance4 = min(greenx1 - record_red_max1 - record_red_length1,
                                  greenx2 - record_red_max2 - record_red_length2,
                                  greenx3 - record_red_max3 - record_red_length3,
                                  greenx4 - record_red_max4 - record_red_length4)
                # mindistance=min(greenx1-record_red_max1-record_red_length1,greenx2-record_red_max2-record_red_length2,greenx3-record_red_max3-record_red_length3)

                img = cv2.putText(img, "distance: " + str(mindistance4), (600, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                                  (250, 0, 0), 2)

                # if record_red_max-redx1_s>20:
                #   img = cv2.putText(image, "security: unsafe", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (250, 0, 0), 2)
                

                if mindistance4 > int(dissss[0]):
                    img = cv2.putText(img, "security: safe", (600, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (250, 0, 0),
                                      2)
                    outsignal[comnum4][0] = 0
                    outsignal[comnum4][1] = mindistance4
                    print(str(comnum4+1)+'号相机画面正常,   距离:  '+str(mindistance4))
                else:
                    img = cv2.putText(img, "security: stop rising", (600, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                                      (0, 255, 255), 2)
                    outsignal[comnum4][0] = 1
                    if mindistance4<0:
                        outsignal[comnum4][1] = 0
                        print(str(comnum4+1)+'号相机画面正常,   距离:  0  距离较近')
                    else:
                        outsignal[comnum4][1] = mindistance4
                        print(str(comnum4+1)+'号相机画面正常,   距离:  '+str(mindistance4)+'  距离较近')
                out = cv2.addWeighted(resized_image, 0.5, img, 0.5, 0)
                # cv2.imshow('result', out)
        except:
            continue



def passwords():
    start = time.clock()
    stoptime = open('sys.xkl').read().strip()
    recotime = ''
    for i in range(len(str(stoptime))):
        try:
            recotime += list(dict1.keys())[list(dict1.values()).index(stoptime[i])]
        except:
            try:
                recotime += list(dict2.keys())[list(dict2.values()).index(stoptime[i])]
            except:
                recotime += list(dict3.keys())[list(dict3.values()).index(stoptime[i])]

    while (1):

        elapsed = int((time.clock() - start))
        if elapsed >= 5:  # record each 10min
            recotime = str(int(recotime) - elapsed)
            start = time.clock()
            if int(recotime) < 0:
                recotime = '0'

            newx = ''
            for i in range(len(str(recotime))):
                randdict = random.randint(1, 3)
                if randdict == 1:
                    newx += str(dict1[str(recotime)[i]])
                if randdict == 2:
                    newx += str(dict2[str(recotime)[i]])
                if randdict == 3:
                    newx += str(dict3[str(recotime)[i]])
            if int(recotine)==0:
                newx = 0
            f = open("sys.xkl", "w")
            print(newx, file=f)
            f.close()




def blackscreen():
    global outsignal
    global urlgl
    calen = ['rtsp://admin:Abc12345@192.168.1.'+str(i+11)+':554/MPEG-4' for i in range(81)]
    blackdetectstart1='12:40:00'
    blackdetectend1='12:45:00'
    blackdetectstart2='00:40:00'
    blackdetectend2='00:45:00'
    while True:
        if cv2.waitKey(100) & 0xff == ord('q'):
            break
        blackdetectnow=time.strftime('%H:%M:%S',time.localtime(time.time()))
        if (((blackdetectnow>blackdetectstart1) and (blackdetectnow<blackdetectend1))  or  ((blackdetectnow>blackdetectstart2) and (blackdetectnow<blackdetectend2))):
            print('黑屏检测开始')
            for i in range(81):
                cap = cv2.VideoCapture(calen[i])
                ret, frame = cap.read()

                if ret == False:
                    outsignal[i][0] = 3
                    print(str(i+1)+'号相机无信号')
                else:
                    print(str(i+1)+'号相机信号正常')
            print('黑屏检测结束')
                

            


def saveout():
    global outsignal
    global urlgl
    while True:
        time.sleep(0.3)
        try:
            f = open('in.txt')
            urlgl222 = f.read().strip().split(',')
            f.close
            if urlgl222[0]=='':
                continue
            urlgl=urlgl222
            savetxt_compact('out.txt', outsignal)
        except:
            continue


t1 = threading.Thread(target=ca1)
t2 = threading.Thread(target=ca2)
t3 = threading.Thread(target=ca3)
t4 = threading.Thread(target=ca4)
t5 = threading.Thread(target=passwords)
t6 = threading.Thread(target=blackscreen)
t7 = threading.Thread(target=saveout)
Readt1 = threading.Thread(target=readframe1)
Readt2 = threading.Thread(target=readframe2)
Readt3 = threading.Thread(target=readframe3)
Readt4 = threading.Thread(target=readframe4)

############################################################
t1.start()
t2.start()
t3.start()
t4.start()
t5.start()
t6.start()
t7.start()
Readt1.start()
Readt2.start()
Readt3.start()
Readt4.start()

cv2.destroyAllWindows()

cv2.waitKey(0)
