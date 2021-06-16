import os,time,cv2, sys, math
import tensorflow as tf
import argparse
import numpy as np

from utils import utils, helpers
from builders import model_builder

parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, default=None, required=False, help='The image you want to predict on. ')
parser.add_argument('--checkpoint_path', type=str, default='./save_check/latest_model_BiSeNet_dataset.ckpt', required=True, help='The path to the latest checkpoint weights for your model.')
parser.add_argument('--crop_height', type=int, default=1024, help='Height of cropped input image to network')
parser.add_argument('--crop_width', type=int, default=1024, help='Width of cropped input image to network')
parser.add_argument('--model', type=str, default='BiSeNet', required=True, help='The model you are using')
parser.add_argument('--dataset', type=str, default="dataset", required=False, help='The dataset you are using')
args = parser.parse_args()

class_names_list, label_values = helpers.get_label_info(os.path.join(args.dataset, "class_dict.csv"))

num_classes = len(label_values)

print("\n***** Begin prediction *****")
print("Dataset -->", args.dataset)
print("Model -->", args.model)
print("Crop Height -->", args.crop_height)
print("Crop Width -->", args.crop_width)
print("Num Classes -->", num_classes)
#print("Image -->", args.image)

# Initializing network
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.Session(config=config)

net_input = tf.placeholder(tf.float32,shape=[None,None,None,3])
net_output = tf.placeholder(tf.float32,shape=[None,None,None,num_classes])

network, _ = model_builder.build_model(args.model, net_input=net_input,
                                        num_classes=num_classes,
                                        crop_width=args.crop_width,
                                        crop_height=args.crop_height,
                                        is_training=False)

sess.run(tf.global_variables_initializer())

print('Loading model checkpoint weights')
saver=tf.train.Saver(max_to_keep=1000)
saver.restore(sess, args.checkpoint_path)


#print("Testing image " + args.image)

cap = cv2.VideoCapture("/home/xuekelou414/Downloads/Camera 05.mp4")#打开视频
size =(args.crop_width,args.crop_height)
#保存视频编码类
video_writer = cv2.VideoWriter(r'/home/xuekelou414/hm/Semantic-Segmentation-Suite-master/video_downloads/outputVideo.avi',
                               cv2.VideoWriter_fourcc(*'XVID'), 30, size)

while cap.isOpened():
    ret, frame = cap.read()

    #cv2.imshow('frame',frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    loaded_image = frame
   # loaded_image = utils.load_image(args.image)
    resized_image =cv2.resize(loaded_image, (args.crop_width, args.crop_height))
    input_image = np.expand_dims(np.float32(resized_image[:args.crop_height, :args.crop_width]),axis=0)/255.0

    st = time.time()
    output_image = sess.run(network,feed_dict={net_input:input_image})

    run_time = time.time()-st

    output_image = np.array(output_image[0,:,:,:])
    output_image = helpers.reverse_one_hot(output_image)

    out_vis_image = helpers.colour_code_segmentation(output_image, label_values)
    file_name = utils.filepath_to_name(args.image)
    img1 = cv2.resize(cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_RGB2BGR), (args.crop_width, args.crop_height),
                      interpolation=cv2.INTER_AREA)
    #cv2.imwrite("%s_pred.png"%(file_name),cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_RGB2BGR))
    cv2.imshow('result', img1)
    # if frame is None:
    #     break
    video_writer.write(img1)
video_writer.release()
# print("")
# print("Finished!")
# print("Wrote image " + "%s_pred.png"%(file_name))
