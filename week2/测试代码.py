##测试代码，生成json结果
# -- coding: utf-8 --
import sys
sys.path.append('/home/ssw/caffe/python')
import matplotlib.pyplot as plt
import os
import caffe
import numpy as np
import pickle
import json
from tqdm import tqdm
root='/home/ssw/Desktop/YFF/faceTask/FER_image/'   #根目录
deploy='/home/ssw/Desktop/YFF/faceTask/FER_image/deploy_alex.prototxt'    #deploy文件绝对路径
caffe_model='/home/ssw/Desktop/YFF/faceTask/train_done/premodel/model_iter__iter_80000.caffemodel'   #训练好的 caffemodel
net = caffe.Net(deploy,caffe_model,caffe.TEST)   #加载model和network


#图片预处理设置
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})  #设定图片的shape格式(1,3,28,28)
transformer.set_transpose('data', (2,0,1))    #改变维度的顺序，由原始图片(28,28,3)变为(3,28,28)
#transformer.set_mean('data', np.load(mean_file).mean(1).mean(1))    #减去均值，前面训练模型时没有减均值，这儿就不用
transformer.set_raw_scale('data', 1)    # 缩放到【0，255】之间
transformer.set_channel_swap('data', (2,1,0))   #交换通道，将图片由RGB变为BGR
#加载label字典，线上不需要，现在需要
# char_dict = pick.load(open('./char_dict','rb'))
#字典键值对换序
src = '/home/ssw/Desktop/YFF/faceTask/test_done/dete2/'
fileName = os.listdir(src)
answer=[]
for imgName in tqdm(fileName):
    img = src+imgName
    im = caffe.io.load_image(img)  # 加载图片
    net.blobs['data'].data[...] = transformer.preprocess('data', im)  # 执行上面设置的图片预处理操作，并将图片载入到blob中
    out = net.forward()
    top_k = net.blobs['prob'].data[0].flatten().argmax()
    save_dict = dict()
    save_dict['label'] = int(top_k)
    save_dict['filename'] = imgName
    r = json.dumps(save_dict)
    answer.append(r)
submit_name = './submit7.json'
with open(submit_name,'w')as f:
    wirte_str = '['+','.join(answer)+']'
    f.write(wirte_str)

