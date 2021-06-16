import os,time,cv2, sys, math
import tensorflow as tf
import argparse
import numpy as np

from utils import  helpers
from builders import model_builder

class_names_list, label_values = helpers.get_label_info(os.path.join('/home/xuekelou414/hm/Semantic-Segmentation-Suite-master/dataset', "class_dict.csv"))

num_classes = len(label_values)


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.Session(config=config)

net_input = tf.placeholder(tf.float32,shape=[None,None,None,3])
net_output = tf.placeholder(tf.float32,shape=[None,None,None,num_classes])

network, _ = model_builder.build_model(model_name="BiSeNet", net_input=net_input,
                                        num_classes=num_classes,
                                        crop_width=1024,
                                        crop_height=1024,
                                        is_training=False)

sess.run(tf.global_variables_initializer())

print('Loading model checkpoint weights')
saver=tf.train.Saver(max_to_keep=1000)
saver.restore(sess,'./Checkpoint/latest_model_BiSeNet_dataset6.ckpt')



cap = cv2.VideoCapture("Camera 06.mp4")


size = (1024, 1024)
# opencv支持不同的编码格式
video_writer = cv2.VideoWriter('outputVideo.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, size)


while cap.isOpened():
    ret, frame = cap.read()
    # cv2.imshow("capture", frame)　　　　　　　　　
    if cv2.waitKey(100) & 0xff == ord('q'):
        break
    loaded_image = frame
    resized_image =cv2.resize(loaded_image, (1024, 1024))
    input_image = np.expand_dims(np.float32(resized_image[:1024,:1024]),axis=0)/255.0

    st = time.time()
    output_image = sess.run(network,feed_dict={net_input:input_image})

    run_time = time.time()-st

    output_image = np.array(output_image[0,:,:,:])
    output_image = helpers.reverse_one_hot(output_image)

    out_vis_image = helpers.colour_code_segmentation(output_image, label_values)
    # file_name = utils.filepath_to_name(args.image)
    #cv2.imwrite("%s_pred.png"%(file_name),cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_RGB2BGR))
    img2 = cv2.resize(cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_RGB2BGR), (1024, 1024), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)  # 转换颜色空间

    ret, thresh = cv2.threshold(gray, 5, 100, 0)   #转化为二值图

    contours, hier = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    img2 = cv2.drawContours(img2, contours, -1, (128, 255, 255), 1)
    gray_1 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # 设置的阈值
    threshold_level = 80

    # 找到低于阈值的所有像素的坐标
    coords = np.column_stack(np.where(gray_1 < threshold_level))
    # 创建一个低于阈值的像素mask
    mask = gray_1 < threshold_level

    # 为mask的像素着色
    img2[mask] = (0, 0, 0)

    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        a = contours[i]
        num1 = len(a)
        b = a[:, 0]
        if area < 10000:
            for j in range(num1):
                c = b[j]
                img2[c[1], c[0]] = (0, 0, 0)

    out=cv2.addWeighted(resized_image, 0.7, img2, 0.3, 0)


    cv2.imshow('result',out)
    if frame is None:
        break
    video_writer.write(out)
video_writer.release()




