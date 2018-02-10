#说明：全部数据在线下测试通过，这里只是拿train的一组数据做测试
import os
os.chdir('/home/kesci/work/face-detection-mtcnn/')#切换目录至MTCNN根目录
from scipy import misc
import tensorflow as tf
import detect_face
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
%pylab inline

minsize = 20 # 最小的人脸尺寸
threshold = [ 0.6, 0.7, 0.7 ]  # MTCNN 三个网络的判断阈值
factor = 0.709 # 尺度因子
gpu_memory_fraction=1.0


print('Creating networks and loading parameters')

with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

            

imgsrc = '/mnt/datasets/train_images/sadness/'     
for imgName in tqdm(os.listdir(imgsrc)):
    image_path = imgsrc+imgName
    print(image_path)
    img = cv2.imread(image_path)
    bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
    nrof_faces = bounding_boxes.shape[0]#人脸数目
    print('找到人脸数目为：{}'.format(nrof_faces))

    # print(bounding_boxes)
    if nrof_faces == 0:
        #如果检测不到人脸则将图片放置在train_notdetected文件夹下
        cv2.imwrite('/mnt/datasets/train/train_notdetected/'+str(imgName),img)
    else:
        #如果检测到人脸则将人脸ROI区域剪裁出后保存到train_detected文件夹下
        crop_faces=[]
        face_position = bounding_boxes[0]
        face_position=face_position.astype(int)
        #     print(face_position[0:4])
        cv2.rectangle(img, (face_position[0], face_position[1]), (face_position[2], face_position[3]), (0, 255, 0), 1)
        crop=img[face_position[1]:face_position[3],
        face_position[0]:face_position[2],]
        crop = cv2.resize(crop, (108, 108), interpolation=cv2.INTER_CUBIC )
        tosrc = ''
        cv2.imwrite('/mnt/datasets/train/train_detected/' + str(imgName), crop)